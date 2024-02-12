/*
// Copyright (C) 2018-2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/detection_model.h>
#include <models/detection_model_centernet.h>
#include <models/detection_model_faceboxes.h>
#include <models/detection_model_retinaface.h>
#include <models/detection_model_retinaface_pt.h>
#include <models/detection_model_ssd.h>
#include <models/detection_model_yolo.h>
#include <models/detection_model_yolov3_onnx.h>
#include <models/detection_model_yolox.h>
#include <models/input_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <monitors/presenter.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/default_flags.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

DEFINE_INPUT_FLAGS
DEFINE_OUTPUT_FLAGS

static const char help_message[] = "Print a usage message.";
static const char at_message[] =
    "Required. Architecture type: centernet, faceboxes, retinaface, retinaface-pytorch, ssd, yolo, yolov3-onnx or yolox";
static const char model_message[] = "Required. Path to an .xml file with a trained model.";
static const char target_device_message[] =
    "Optional. Specify the target device to infer on (the list of available devices is shown below). "
    "Default value is CPU. Use \"-d HETERO:<comma-separated_devices_list>\" format to specify HETERO plugin. "
    "The demo will look for a suitable plugin for a specified device.";
static const char labels_message[] = "Optional. Path to a file with labels mapping.";
static const char layout_message[] = "Optional. Specify inputs layouts."
                                     " Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.";
static const char thresh_output_message[] = "Optional. Probability threshold for detections.";
static const char raw_output_message[] = "Optional. Inference results as raw values.";
static const char input_resizable_message[] =
    "Optional. Enables resizable input with support of ROI crop & auto resize.";
static const char nireq_message[] = "Optional. Number of infer requests. If this option is omitted, number of infer "
                                    "requests is determined automatically.";
static const char num_threads_message[] = "Optional. Number of threads.";
static const char num_streams_message[] = "Optional. Number of streams to use for inference on the CPU or/and GPU in "
                                          "throughput mode (for HETERO and MULTI device cases use format "
                                          "<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)";
static const char no_show_message[] = "Optional. Don't show output.";
static const char utilization_monitors_message[] = "Optional. List of monitors to show initially.";
static const char iou_thresh_output_message[] =
    "Optional. Filtering intersection over union threshold for overlapping boxes.";
static const char yolo_af_message[] = "Optional. Use advanced postprocessing/filtering algorithm for YOLO.";
static const char output_resolution_message[] =
    "Optional. Specify the maximum output window resolution "
    "in (width x height) format. Example: 1280x720. Input frame size used by default.";
static const char anchors_message[] = "Optional. A comma separated list of anchors. "
                                      "By default used default anchors for model. Only for YOLOV4 architecture type.";
static const char masks_message[] = "Optional. A comma separated list of mask for anchors. "
                                    "By default used default masks for model. Only for YOLOV4 architecture type.";
static const char reverse_input_channels_message[] = "Optional. Switch the input channels order from BGR to RGB.";
static const char mean_values_message[] =
    "Optional. Normalize input by subtracting the mean values per channel. Example: \"255.0 255.0 255.0\"";
static const char scale_values_message[] = "Optional. Divide input by scale values per channel. Division is applied "
                                           "after mean values subtraction. Example: \"255.0 255.0 255.0\"";

DEFINE_bool(h, false, help_message);
DEFINE_string(at, "", at_message);
DEFINE_string(m, "", model_message);
DEFINE_string(d, "CPU", target_device_message);
DEFINE_string(labels, "", labels_message);
DEFINE_string(layout, "", layout_message);
DEFINE_bool(r, false, raw_output_message);
DEFINE_double(t, 0.5, thresh_output_message);
DEFINE_double(iou_t, 0.5, iou_thresh_output_message);
DEFINE_bool(auto_resize, false, input_resizable_message);
DEFINE_uint32(nireq, 0, nireq_message);
DEFINE_uint32(nthreads, 0, num_threads_message);
DEFINE_string(nstreams, "", num_streams_message);
DEFINE_bool(no_show, false, no_show_message);
DEFINE_string(u, "", utilization_monitors_message);
DEFINE_bool(yolo_af, true, yolo_af_message);
DEFINE_string(output_resolution, "", output_resolution_message);
DEFINE_string(anchors, "", anchors_message);
DEFINE_string(masks, "", masks_message);
DEFINE_bool(reverse_input_channels, false, reverse_input_channels_message);
DEFINE_string(mean_values, "", mean_values_message);
DEFINE_string(scale_values, "", scale_values_message);

/**
 * \brief This function shows a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "object_detection_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -at \"<type>\"              " << at_message << std::endl;
    std::cout << "    -i                        " << input_message << std::endl;
    std::cout << "    -m \"<path>\"               " << model_message << std::endl;
    std::cout << "    -o \"<path>\"               " << output_message << std::endl;
    std::cout << "    -limit \"<num>\"            " << limit_message << std::endl;
    std::cout << "    -d \"<device>\"             " << target_device_message << std::endl;
    std::cout << "    -labels \"<path>\"          " << labels_message << std::endl;
    std::cout << "    -layout \"<string>\"        " << layout_message << std::endl;
    std::cout << "    -r                        " << raw_output_message << std::endl;
    std::cout << "    -t                        " << thresh_output_message << std::endl;
    std::cout << "    -iou_t                    " << iou_thresh_output_message << std::endl;
    std::cout << "    -auto_resize              " << input_resizable_message << std::endl;
    std::cout << "    -nireq \"<integer>\"        " << nireq_message << std::endl;
    std::cout << "    -nthreads \"<integer>\"     " << num_threads_message << std::endl;
    std::cout << "    -nstreams                 " << num_streams_message << std::endl;
    std::cout << "    -loop                     " << loop_message << std::endl;
    std::cout << "    -no_show                  " << no_show_message << std::endl;
    std::cout << "    -output_resolution        " << output_resolution_message << std::endl;
    std::cout << "    -u                        " << utilization_monitors_message << std::endl;
    std::cout << "    -yolo_af                  " << yolo_af_message << std::endl;
    std::cout << "    -anchors                  " << anchors_message << std::endl;
    std::cout << "    -masks                    " << masks_message << std::endl;
    std::cout << "    -reverse_input_channels   " << reverse_input_channels_message << std::endl;
    std::cout << "    -mean_values              " << mean_values_message << std::endl;
    std::cout << "    -scale_values             " << scale_values_message << std::endl;
}

class ColorPalette {
private:
    std::vector<cv::Scalar> palette;

    static double getRandom(double a = 0.0, double b = 1.0) {
        static std::default_random_engine e;
        std::uniform_real_distribution<> dis(a, std::nextafter(b, std::numeric_limits<double>::max()));
        return dis(e);
    }

    static double distance(const cv::Scalar& c1, const cv::Scalar& c2) {
        auto dh = std::fmin(std::fabs(c1[0] - c2[0]), 1 - fabs(c1[0] - c2[0])) * 2;
        auto ds = std::fabs(c1[1] - c2[1]);
        auto dv = std::fabs(c1[2] - c2[2]);

        return dh * dh + ds * ds + dv * dv;
    }

    static cv::Scalar maxMinDistance(const std::vector<cv::Scalar>& colorSet,
                                     const std::vector<cv::Scalar>& colorCandidates) {
        std::vector<double> distances;
        distances.reserve(colorCandidates.size());
        for (auto& c1 : colorCandidates) {
            auto min =
                *std::min_element(colorSet.begin(), colorSet.end(), [&c1](const cv::Scalar& a, const cv::Scalar& b) {
                    return distance(c1, a) < distance(c1, b);
                });
            distances.push_back(distance(c1, min));
        }
        auto max = std::max_element(distances.begin(), distances.end());
        return colorCandidates[std::distance(distances.begin(), max)];
    }

    static cv::Scalar hsv2rgb(const cv::Scalar& hsvColor) {
        cv::Mat rgb;
        cv::Mat hsv(1, 1, CV_8UC3, hsvColor);
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
        return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
    }

public:
    explicit ColorPalette(size_t n) {
        palette.reserve(n);
        std::vector<cv::Scalar> hsvColors(1, {1., 1., 1.});
        std::vector<cv::Scalar> colorCandidates;
        size_t numCandidates = 100;

        hsvColors.reserve(n);
        colorCandidates.resize(numCandidates);
        for (size_t i = 1; i < n; ++i) {
            std::generate(colorCandidates.begin(), colorCandidates.end(), []() {
                return cv::Scalar{getRandom(), getRandom(0.8, 1.0), getRandom(0.5, 1.0)};
            });
            hsvColors.push_back(maxMinDistance(hsvColors, colorCandidates));
        }

        for (auto& hsv : hsvColors) {
            // Convert to OpenCV HSV format
            hsv[0] *= 179;
            hsv[1] *= 255;
            hsv[2] *= 255;

            palette.push_back(hsv2rgb(hsv));
        }
    }

    const cv::Scalar& operator[](size_t index) const {
        return palette[index % palette.size()];
    }
};

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_at.empty()) {
        throw std::logic_error("Parameter -at is not set");
    }

    if (!FLAGS_output_resolution.empty() && FLAGS_output_resolution.find("x") == std::string::npos) {
        throw std::logic_error("Correct format of -output_resolution parameter is \"width\"x\"height\".");
    }
    return true;
}

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(DetectionResult& result, const ColorPalette& palette, OutputTransform& outputTransform) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<ImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    outputTransform.resize(outputImg);
    // Visualizing result data over source image
    if (FLAGS_r) {
        slog::debug << " -------------------- Frame # " << result.frameId << "--------------------" << slog::endl;
        slog::debug << " Class ID  | Confidence | XMIN | YMIN | XMAX | YMAX " << slog::endl;
    }

    for (auto& obj : result.objects) {
        if (FLAGS_r) {
            slog::debug << " " << std::left << std::setw(9) << obj.label << " | " << std::setw(10) << obj.confidence
                        << " | " << std::setw(4) << int(obj.x) << " | " << std::setw(4) << int(obj.y) << " | "
                        << std::setw(4) << int(obj.x + obj.width) << " | " << std::setw(4) << int(obj.y + obj.height)
                        << slog::endl;
        }
        outputTransform.scaleRect(obj);
        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(1) << obj.confidence * 100 << '%';
        const auto& color = palette[obj.labelID];
        putHighlightedText(outputImg,
                           obj.label + conf.str(),
                           cv::Point2f(obj.x, obj.y - 5),
                           cv::FONT_HERSHEY_COMPLEX_SMALL,
                           1,
                           color,
                           2);
        cv::rectangle(outputImg, obj, color, 2);
    }

    try {
        for (auto& lmark : result.asRef<RetinaFaceDetectionResult>().landmarks) {
            outputTransform.scaleCoord(lmark);
            cv::circle(outputImg, lmark, 2, cv::Scalar(0, 255, 255), -1);
        }
    } catch (const std::bad_cast&) {}

    return outputImg;
}

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        const auto& strAnchors = split(FLAGS_anchors, ',');
        const auto& strMasks = split(FLAGS_masks, ',');

        std::vector<float> anchors;
        std::vector<int64_t> masks;
        try {
            for (auto& str : strAnchors) {
                anchors.push_back(std::stof(str));
            }
        } catch (...) { throw std::runtime_error("Invalid anchors list is provided."); }

        try {
            for (auto& str : strMasks) {
                masks.push_back(std::stoll(str));
            }
        } catch (...) { throw std::runtime_error("Invalid masks list is provided."); }

        //------------------------------- Preparing Input ------------------------------------------------------
        auto cap = openImagesCapture(FLAGS_i, FLAGS_loop, FLAGS_nireq == 1 ? read_type::efficient : read_type::safe);
        cv::Mat curr_frame;

        //------------------------------ Running Detection routines ----------------------------------------------
        std::vector<std::string> labels;
        if (!FLAGS_labels.empty())
            labels = DetectionModel::loadLabels(FLAGS_labels);
        ColorPalette palette(labels.size() > 0 ? labels.size() : 100);

        std::unique_ptr<ModelBase> model;
        if (FLAGS_at == "centernet") {
            model.reset(new ModelCenterNet(FLAGS_m, static_cast<float>(FLAGS_t), labels, FLAGS_layout));
        } else if (FLAGS_at == "faceboxes") {
            model.reset(new ModelFaceBoxes(FLAGS_m,
                                           static_cast<float>(FLAGS_t),
                                           FLAGS_auto_resize,
                                           static_cast<float>(FLAGS_iou_t),
                                           FLAGS_layout));
        } else if (FLAGS_at == "retinaface") {
            model.reset(new ModelRetinaFace(FLAGS_m,
                                            static_cast<float>(FLAGS_t),
                                            FLAGS_auto_resize,
                                            static_cast<float>(FLAGS_iou_t),
                                            FLAGS_layout));
        } else if (FLAGS_at == "retinaface-pytorch") {
            model.reset(new ModelRetinaFacePT(FLAGS_m,
                                              static_cast<float>(FLAGS_t),
                                              FLAGS_auto_resize,
                                              static_cast<float>(FLAGS_iou_t),
                                              FLAGS_layout));
        } else if (FLAGS_at == "ssd") {
            model.reset(new ModelSSD(FLAGS_m, static_cast<float>(FLAGS_t), FLAGS_auto_resize, labels, FLAGS_layout));
        } else if (FLAGS_at == "yolo") {
            model.reset(new ModelYolo(FLAGS_m,
                                      static_cast<float>(FLAGS_t),
                                      FLAGS_auto_resize,
                                      FLAGS_yolo_af,
                                      static_cast<float>(FLAGS_iou_t),
                                      labels,
                                      anchors,
                                      masks,
                                      FLAGS_layout));
        } else if (FLAGS_at == "yolov3-onnx") {
            model.reset(new ModelYoloV3ONNX(FLAGS_m,
                                            static_cast<float>(FLAGS_t),
                                            labels,
                                            FLAGS_layout));
        } else if (FLAGS_at == "yolox") {
            model.reset(new ModelYoloX(FLAGS_m,
                                       static_cast<float>(FLAGS_t),
                                       static_cast<float>(FLAGS_iou_t),
                                       labels,
                                       FLAGS_layout));
        } else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }
        model->setInputsPreprocessing(FLAGS_reverse_input_channels, FLAGS_mean_values, FLAGS_scale_values);
        slog::info << ov::get_openvino_version() << slog::endl;

        ov::Core core;

        AsyncPipeline pipeline(std::move(model),
                               ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads),
                               core);
        Presenter presenter(FLAGS_u);

        bool keepRunning = true;
        int64_t frameNum = -1;
        std::unique_ptr<ResultBase> result;
        uint32_t framesProcessed = 0;

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        PerformanceMetrics renderMetrics;

        cv::Size outputResolution;
        OutputTransform outputTransform = OutputTransform();
        size_t found = FLAGS_output_resolution.find("x");

        while (keepRunning) {
            if (pipeline.isReadyToProcess()) {
                auto startTime = std::chrono::steady_clock::now();

                //--- Capturing frame
                curr_frame = cap->read();

                if (curr_frame.empty()) {
                    // Input stream is over
                    break;
                }

                frameNum = pipeline.submitData(ImageInputData(curr_frame),
                                               std::make_shared<ImageMetaData>(curr_frame, startTime));
            }

            if (frameNum == 0) {
                if (found == std::string::npos) {
                    outputResolution = curr_frame.size();
                } else {
                    outputResolution = cv::Size{
                        std::stoi(FLAGS_output_resolution.substr(0, found)),
                        std::stoi(FLAGS_output_resolution.substr(found + 1, FLAGS_output_resolution.length()))};
                    outputTransform = OutputTransform(curr_frame.size(), outputResolution);
                    outputResolution = outputTransform.computeResolution();
                }
            }

            //--- Waiting for free input slot or output data available. Function will return immediately if any of them
            // are available.
            pipeline.waitForData();

            //--- Checking for results and rendering data if it's ready
            //--- If you need just plain data without rendering - cast result's underlying pointer to DetectionResult*
            //    and use your own processing instead of calling renderDetectionData().
            while (keepRunning && (result = pipeline.getResult())) {
                auto renderingStart = std::chrono::steady_clock::now();
                cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), palette, outputTransform);

                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                renderMetrics.update(renderingStart);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);

                videoWriter.write(outFrame);
                framesProcessed++;

                if (!FLAGS_no_show) {
                    cv::imshow("Detection Results", outFrame);
                    //--- Processing keyboard events
                    int key = cv::waitKey(1);
                    if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                        keepRunning = false;
                    } else {
                        presenter.handleKey(key);
                    }
                }
            }
        }  // while(keepRunning)

        // ------------ Waiting for completion of data processing and rendering the rest of results ---------
        pipeline.waitForTotalCompletion();

        for (; framesProcessed <= frameNum; framesProcessed++) {
            result = pipeline.getResult();
            if (result != nullptr) {
                auto renderingStart = std::chrono::steady_clock::now();
                cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), palette, outputTransform);
                //--- Showing results and device information
                presenter.drawGraphs(outFrame);
                renderMetrics.update(renderingStart);
                metrics.update(result->metaData->asRef<ImageMetaData>().timeStamp,
                               outFrame,
                               {10, 22},
                               cv::FONT_HERSHEY_COMPLEX,
                               0.65);
                videoWriter.write(outFrame);
                if (!FLAGS_no_show) {
                    cv::imshow("Detection Results", outFrame);
                    //--- Updating output window
                    cv::waitKey(1);
                }
            }
        }

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        logLatencyPerStage(cap->getMetrics().getTotal().latency,
                           pipeline.getPreprocessMetrics().getTotal().latency,
                           pipeline.getInferenceMetircs().getTotal().latency,
                           pipeline.getPostprocessMetrics().getTotal().latency,
                           renderMetrics.getTotal().latency);
        slog::info << presenter.reportMeans() << slog::endl;
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
