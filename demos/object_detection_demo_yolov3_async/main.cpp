// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_yolov3_async/main.cpp
* \example object_detection_demo_yolov3_async/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include <monitors/presenter.h>
#include <samples/images_capture.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "object_detection_demo_yolov3_async.hpp"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating the input arguments--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    return true;
}

void FrameToBlob(const cv::Mat &frame, InferRequest::Ptr &inferRequest, const std::string &inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) {
        this->xmin = static_cast<int>((x - w / 2) * w_scale);
        this->ymin = static_cast<int>((y - h / 2) * h_scale);
        this->xmax = static_cast<int>(this->xmin + w * w_scale);
        this->ymax = static_cast<int>(this->ymin + h * h_scale);
        this->class_id = class_id;
        this->confidence = confidence;
    }

    bool operator <(const DetectionObject &s2) const {
        return this->confidence < s2.confidence;
    }
    bool operator >(const DetectionObject &s2) const {
        return this->confidence > s2.confidence;
    }
};

double IntersectionOverUnion(const DetectionObject &box_1, const DetectionObject &box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

class YoloParams {
    template <typename T>
    void computeAnchors(const std::vector<T> & mask) {
        std::vector<float> maskedAnchors(num * 2);
        for (int i = 0; i < num; ++i) {
            maskedAnchors[i * 2] = anchors[mask[i] * 2];
            maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
        }
        anchors = maskedAnchors;
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0,
                                  156.0, 198.0, 373.0, 326.0};

    YoloParams() {}

    YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
        coords = regionYolo->get_num_coords();
        classes = regionYolo->get_num_classes();
        anchors = regionYolo->get_anchors();
        auto mask = regionYolo->get_mask();
        num = mask.size();

        computeAnchors(mask);
    }
};

void ParseYOLOV3Output(const YoloParams &params, const std::string & output_name,
                       const Blob::Ptr &blob, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + output_name +
        " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));

    auto side = out_blob_h;
    auto side_square = side * side;
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < params.num; ++n) {
            int obj_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
            int box_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * params.anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * params.anchors[2 * n];
            for (int j = 0; j < params.classes; ++j) {
                int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                        static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                        static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}


int main(int argc, char *argv[]) {
    try {
        /** This demo covers a certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validating the input arguments ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        /**Loading extensions to the devices **/

        if (!FLAGS_l.empty()) {
            // CPU extensions are loaded as a shared library and passed as a pointer to the base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            ie.AddExtension(extension_ptr, "CPU");
        }
        if (!FLAGS_c.empty()) {
            // GPU extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
        }

        /** Per-layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------- 2. Reading the IR generated by the Model Optimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        /** Reading network model **/
        auto cnnNetwork = ie.ReadNetwork(FLAGS_m);
        /** Reading labels (if specified) **/
        std::vector<std::string> labels;
        if (!FLAGS_labels.empty()) {
            std::ifstream inputFile(FLAGS_labels);
            std::string label; 
            while (std::getline(inputFile, label)) {
                labels.push_back(label);
            }
            if (labels.empty())
                throw std::logic_error("File empty or not found: " + FLAGS_labels);
        }
        // -----------------------------------------------------------------------------------------------------

        /** YOLOV3-based network should have one input and three output **/
        // --------------------------- 3. Configuring input and output -----------------------------------------
        // --------------------------------- Preparing input blobs ---------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks that have only one input");
        }
        InputInfo::Ptr& input = inputInfo.begin()->second;
        auto inputName = inputInfo.begin()->first;
        input->setPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            input->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            input->getInputData()->setLayout(Layout::NHWC);
        } else {
            input->getInputData()->setLayout(Layout::NCHW);
        }

        ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
        SizeVector& inSizeVector = inputShapes.begin()->second;
        inSizeVector[0] = 1;  // set batch to 1
        cnnNetwork.reshape(inputShapes);
        // --------------------------------- Preparing output blobs -------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
        for (auto &output : outputInfo) {
            output.second->setPrecision(Precision::FP32);
            output.second->setLayout(Layout::NCHW);
        }

        std::map<std::string, YoloParams> yoloParams;
        if (auto ngraphFunction = cnnNetwork.getFunction()) {
            for (const auto op : ngraphFunction->get_ops()) {
                auto outputLayer = outputInfo.find(op->get_friendly_name());
                if (outputLayer != outputInfo.end()) {
                    auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                    if (!regionYolo) {
                        throw std::runtime_error("Invalid output type: " +
                            std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                    }
                    yoloParams[outputLayer->first] = YoloParams(regionYolo);
                }
            }
        }
        else {
            throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
        }

        if (!labels.empty() && static_cast<int>(labels.size()) != yoloParams.begin()->second.classes) {
            throw std::runtime_error("The number of labels is different from numbers of model classes");
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork network = ie.LoadNetwork(cnnNetwork, FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Creating infer request -----------------------------------------------
        InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Doing inference ------------------------------------------------------
        slog::info << "Start inference " << slog::endl;

        bool isLastFrame = false;
        bool isAsyncMode = false;  // execution is always started using SYNC mode
        bool isModeChanged = false;  // set to TRUE when execution mode is changed (SYNC<->ASYNC)

        cv::Mat frame = cap->read();
        if (!frame.data) throw std::runtime_error("Can't read an image from the input");

        cv::Size graphSize{frame.cols / 4, 60};
        Presenter presenter(FLAGS_u, frame.rows - graphSize.height - 10, graphSize);

        std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key" << std::endl;
        std::cout << "To switch between sync/async modes, press TAB key in the output window" << std::endl;

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double ocv_render_time = 0;
        while (true) {
            auto t0 = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the Async mode, we capture frame to populate the NEXT infer request
            // in the regular mode, we capture frame to the CURRENT infer request
            cv::Mat next_frame = cap->read();
            if (!next_frame.data) {
                isLastFrame = true;  // end of video file
            }
            if (isAsyncMode) {
                if (isModeChanged) {
                    FrameToBlob(frame, async_infer_request_curr, inputName);
                }
                if (!isLastFrame) {
                    FrameToBlob(next_frame, async_infer_request_next, inputName);
                }
            } else if (!isModeChanged) {
                FrameToBlob(frame, async_infer_request_curr, inputName);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the true Async mode, we start the NEXT infer request while waiting for the CURRENT to complete
            // in the regular mode, we start the CURRENT request and wait for its completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    async_infer_request_curr->StartAsync();
                }
                if (!isLastFrame) {
                    async_infer_request_next->StartAsync();
                }
            } else if (!isModeChanged) {
                async_infer_request_curr->StartAsync();
            }

            if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                t1 = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1 - t0);

                t0 = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                t0 = std::chrono::high_resolution_clock::now();
                presenter.drawGraphs(frame);
                std::ostringstream out;
                out << "OpenCV cap/render time: " << std::fixed << std::setprecision(2)
                    << (ocv_decode_time + ocv_render_time) << " ms";
                cv::putText(frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 255, 0));
                out.str("");
                out << "Wallclock time " << (isAsyncMode ? "(TRUE ASYNC):      " : "(SYNC, press Tab): ");
                out << std::fixed << std::setprecision(2) << wall.count() << " ms (" << 1000.f / wall.count() << " fps)";
                cv::putText(frame, out.str(), cv::Point2f(0, 50), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));
                if (!isAsyncMode) {  // In the true async mode, there is no way to measure detection time directly
                    out.str("");
                    out << "Detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                        << " ms ("
                        << 1000.f / detection.count() << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 75), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                                cv::Scalar(255, 0, 0));
                }

                // ---------------------------Processing output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                const TensorDesc& inputDesc = inputInfo.begin()->second.get()->getTensorDesc();
                unsigned long resized_im_h = getTensorHeight(inputDesc);
                unsigned long resized_im_w = getTensorWidth(inputDesc);
                std::vector<DetectionObject> objects;
                // Parsing outputs
                for (auto &output : outputInfo) {
                    auto output_name = output.first;
                    Blob::Ptr blob = async_infer_request_curr->GetBlob(output_name);
                    ParseYOLOV3Output(yoloParams[output_name], output_name, blob, resized_im_h, resized_im_w, frame.rows, frame.cols, FLAGS_t, objects);
                }
                // Filtering overlapping boxes
                std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
                for (size_t i = 0; i < objects.size(); ++i) {
                    if (objects[i].confidence == 0)
                        continue;
                    for (size_t j = i + 1; j < objects.size(); ++j)
                        if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t)
                            objects[j].confidence = 0;
                }
                // Drawing boxes
                for (auto &object : objects) {
                    if (object.confidence < FLAGS_t)
                        continue;
                    auto label = object.class_id;
                    float confidence = object.confidence;
                    if (FLAGS_r) {
                        std::cout << "[" << label << "] element, prob = " << confidence <<
                                  "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                                  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                    }
                    if (confidence > FLAGS_t) {
                        /** Drawing only objects when >confidence_threshold probability **/
                        std::ostringstream conf;
                        conf << ":" << std::fixed << std::setprecision(3) << confidence;
                        cv::putText(frame,
                                    (!labels.empty() ? labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                                    cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    cv::Scalar(0, 0, 255));
                        cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                                      cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));
                    }
                }
            }
            if (!FLAGS_no_show) {
                cv::imshow("Detection results", frame);
            }

            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }

            // Final point:
            // in the truly Async mode, we swap the NEXT and CURRENT requests for the next iteration
            frame = next_frame;
            if (isAsyncMode) {
                async_infer_request_curr.swap(async_infer_request_next);
            }

            const int key = cv::waitKey(1);
            if (27 == key)  // Esc
                break;
            if (9 == key) {  // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            } else {
                presenter.handleKey(key);
            }
        }
        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;

        /** Showing performace results **/
        if (FLAGS_pc) {
            printPerformanceCounts(*async_infer_request_curr, std::cout, getFullDeviceName(ie, FLAGS_d));
        }

        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
