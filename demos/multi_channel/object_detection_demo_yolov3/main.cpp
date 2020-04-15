// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/**
* \brief The entry point for the Inference Engine multichannel_yolo_detection demo application
* \file multichannel_yolo_detection/main.cpp
* \example multichannel_yolo_detection/main.cpp
*/
#include <iostream>
#include <vector>
#include <utility>

#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <queue>
#include <chrono>
#include <sstream>
#include <memory>
#include <string>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

#include <opencv2/opencv.hpp>
#include <ngraph/ngraph.hpp>

#include <monitors/presenter.h>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "input.hpp"
#include "multichannel_params.hpp"
#include "multichannel_object_detection_demo_yolov3_params.hpp"
#include "output.hpp"
#include "threading.hpp"
#include "graph.hpp"

namespace {

/**
* \brief This function show a help message
*/
void showUsage() {
    std::cout << std::endl;
    std::cout << "multi_channel_yolo_v3_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -m \"<path>\"                  " << model_path_message<< std::endl;
    std::cout << "      -l \"<absolute_path>\"       " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"       " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"                " << target_device_message << std::endl;
    std::cout << "    -nc                          " << num_cameras << std::endl;
    std::cout << "    -bs                          " << batch_size << std::endl;
    std::cout << "    -nireq                       " << num_infer_requests << std::endl;
    std::cout << "    -n_iqs                       " << input_queue_size << std::endl;
    std::cout << "    -fps_sp                      " << fps_sampling_period << std::endl;
    std::cout << "    -n_sp                        " << num_sampling_periods << std::endl;
    std::cout << "    -pc                          " << performance_counter_message << std::endl;
    std::cout << "    -t                           " << thresh_output_message << std::endl;
    std::cout << "    -no_show                     " << no_show_processed_video << std::endl;
    std::cout << "    -show_stats                  " << show_statistics << std::endl;
    std::cout << "    -duplicate_num               " << duplication_channel_number << std::endl;
    std::cout << "    -real_input_fps              " << real_input_fps << std::endl;
    std::cout << "    -i                           " << input_video << std::endl;
    std::cout << "    -loop_video                  " << loop_video_output_message << std::endl;
    std::cout << "    -u                           " << utilization_monitors_message << std::endl;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    if (FLAGS_nc == 0 && FLAGS_i.empty()) {
        throw std::logic_error("Please specify at least one video source(web cam or video file)");
    }
    slog::info << "\tDetection model:           " << FLAGS_m << slog::endl;
    slog::info << "\tDetection threshold:       " << FLAGS_t << slog::endl;
    slog::info << "\tUtilizing device:          " << FLAGS_d << slog::endl;
    if (!FLAGS_l.empty()) {
        slog::info << "\tCPU extension library:     " << FLAGS_l << slog::endl;
    }
    if (!FLAGS_c.empty()) {
        slog::info << "\tCLDNN custom kernels map:  " << FLAGS_c << slog::endl;
    }
    slog::info << "\tBatch size:                " << FLAGS_bs << slog::endl;
    slog::info << "\tNumber of infer requests:  " << FLAGS_nireq << slog::endl;
    slog::info << "\tNumber of input web cams:  "  << FLAGS_nc << slog::endl;

    return true;
}

static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

class YoloParams {
    template <typename T>
    void computeAnchors(const std::vector<float> & initialAnchors, const std::vector<T> & mask) {
        anchors.resize(num * 2);
        for (int i = 0; i < num; ++i) {
            anchors[i * 2] = initialAnchors[mask[i] * 2];
            anchors[i * 2 + 1] = initialAnchors[mask[i] * 2 + 1];
        }
    }

public:
    int num = 0, classes = 0, coords = 0;
    std::vector<float> anchors;

    YoloParams() {}

    YoloParams(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
        coords = regionYolo->get_num_coords();
        classes = regionYolo->get_num_classes();
        auto initialAnchors = regionYolo->get_anchors();
        auto mask = regionYolo->get_mask();
        num = mask.size();

        computeAnchors(initialAnchors, mask);
    }

    YoloParams(InferenceEngine::CNNLayer::Ptr layer) {
        if (layer->type != "RegionYolo")
            throw std::runtime_error("Invalid output type: " + layer->type + ". RegionYolo expected");

        coords = layer->GetParamAsInt("coords");
        classes = layer->GetParamAsInt("classes");
        auto initialAnchors = layer->GetParamAsFloats("anchors");
        auto mask = layer->GetParamAsInts("mask");
        num = mask.size();

        computeAnchors(initialAnchors, mask);
    }
};

struct DetectionObject {
    int xmin, ymin, xmax, ymax, class_id;
    float confidence;

    DetectionObject(double x, double y, double h, double w, int class_id, float confidence, float h_scale, float w_scale) :
        xmin{static_cast<int>((x - w / 2) * w_scale)},
        ymin{static_cast<int>((y - h / 2) * h_scale)},
        xmax{static_cast<int>(this->xmin + w * w_scale)},
        ymax{static_cast<int>(this->ymin + h * h_scale)},
        class_id{class_id},
        confidence{confidence} {}

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

void ParseYOLOV3Output(InferenceEngine::InferRequest::Ptr req,
                       const std::string &outputName,
                       const YoloParams &yoloParams, const unsigned long resized_im_h,
                       const unsigned long resized_im_w, const unsigned long original_im_h,
                       const unsigned long original_im_w,
                       const double threshold, std::vector<DetectionObject> &objects) {
    InferenceEngine::Blob::Ptr blob = req->GetBlob(outputName);

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output. It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
        ", current W = " + std::to_string(out_blob_h));

    auto num = yoloParams.num;
    auto coords = yoloParams.coords;
    auto classes = yoloParams.classes;

    auto anchors = yoloParams.anchors;

    auto side = out_blob_h;
    auto side_square = side * side;
    const float *output_blob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type *>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < num; ++n) {
            int obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords);
            int box_index = EntryIndex(side, coords, classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * anchors[2 * n];
            for (int j = 0; j < classes; ++j) {
                int class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j);
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

void drawDetections(cv::Mat& img, const std::vector<DetectionObject>& detections, const std::vector<cv::Scalar>& colors) {
    for (const DetectionObject& f : detections) {
        cv::rectangle(img,
                      cv::Rect2f(static_cast<float>(f.xmin),
                                 static_cast<float>(f.ymin),
                                 static_cast<float>((f.xmax-f.xmin)),
                                 static_cast<float>((f.ymax-f.ymin))),
                      colors[static_cast<int>(f.class_id)],
                      2);
    }
}

const size_t DISP_WIDTH  = 1920;
const size_t DISP_HEIGHT = 1080;
const size_t MAX_INPUTS  = 25;

struct DisplayParams {
    std::string name;
    cv::Size windowSize;
    cv::Size frameSize;
    size_t count;
    cv::Point points[MAX_INPUTS];
};

DisplayParams prepareDisplayParams(size_t count) {
    DisplayParams params;
    params.count = count;
    params.windowSize = cv::Size(DISP_WIDTH, DISP_HEIGHT);

    size_t gridCount = static_cast<size_t>(ceil(sqrt(count)));
    size_t gridStepX = static_cast<size_t>(DISP_WIDTH/gridCount);
    size_t gridStepY = static_cast<size_t>(DISP_HEIGHT/gridCount);
    params.frameSize = cv::Size(gridStepX, gridStepY);

    for (size_t i = 0; i < count; i++) {
        cv::Point p;
        p.x = gridStepX * (i/gridCount);
        p.y = gridStepY * (i%gridCount);
        params.points[i] = p;
    }
    return params;
}

std::map<std::string, YoloParams> GetYoloParams(const std::vector<std::string>& outputDataBlobNames,
                                                InferenceEngine::CNNNetwork &network) {
    std::map<std::string, YoloParams> __yoloParams;

    for (auto &output_name : outputDataBlobNames) {
        YoloParams params;

        if (auto ngraphFunction = network.getFunction()) {
            for (const auto op : ngraphFunction->get_ops()) {
                if (op->get_friendly_name() == output_name) {
                    auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                    if (!regionYolo) {
                        throw std::runtime_error("Invalid output type: " +
                            std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                    }

                    params = regionYolo;
                    break;
                }
            }
        } else {
            throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
        }
        __yoloParams.insert(std::pair<std::string, YoloParams>(output_name.c_str(), params));
    }

    return __yoloParams;
}

void displayNSources(const std::vector<std::shared_ptr<VideoFrame>>& data,
                     float time,
                     const std::string& stats,
                     const DisplayParams& params,
                     const std::vector<cv::Scalar> &colors,
                     Presenter& presenter) {
    cv::Mat windowImage = cv::Mat::zeros(params.windowSize, CV_8UC3);
    auto loopBody = [&](size_t i) {
        auto& elem = data[i];
        if (!elem->frame.empty()) {
            cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
            cv::Mat windowPart = windowImage(rectFrame);
            cv::resize(elem->frame, windowPart, params.frameSize);
            drawDetections(windowPart, elem->detections.get<std::vector<DetectionObject>>(), colors);
        }
    };

    auto drawStats = [&]() {
        if (FLAGS_show_stats && !stats.empty()) {
            static const cv::Point posPoint = cv::Point(3*DISP_WIDTH/4, 4*DISP_HEIGHT/5);
            auto pos = posPoint + cv::Point(0, 25);
            size_t currPos = 0;
            while (true) {
                auto newPos = stats.find('\n', currPos);
                cv::putText(windowImage, stats.substr(currPos, newPos - currPos), pos, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8,  cv::Scalar(0, 0, 255), 1);
                if (newPos == std::string::npos) {
                    break;
                }
                pos += cv::Point(0, 25);
                currPos = newPos + 1;
            }
        }
    };

//  #ifdef USE_TBB
#if 0  // disable multithreaded rendering for now
    run_in_arena([&](){
        tbb::parallel_for<size_t>(0, data.size(), [&](size_t i) {
            loopBody(i);
        });
    });
#else
    for (size_t i = 0; i < data.size(); ++i) {
        loopBody(i);
    }
#endif
    presenter.drawGraphs(windowImage);
    drawStats();

    char str[256];
    snprintf(str, sizeof(str), "%5.2f fps", static_cast<double>(1000.0f/time));
    cv::putText(windowImage, str, cv::Point(800, 100), cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 2.0,  cv::Scalar(0, 255, 0), 2);
    cv::imshow(params.name, windowImage);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
#if USE_TBB
        TbbArenaWrapper arena;
#endif

        slog::info << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::string modelPath = FLAGS_m;
        std::size_t found = modelPath.find_last_of(".");
        if (found > modelPath.size()) {
            slog::info << "Invalid model name: " << modelPath << slog::endl;
            slog::info << "Expected to be <model_name>.xml" << slog::endl;
            return -1;
        }
        slog::info << "Model   path: " << modelPath << slog::endl;

        std::map<std::string, YoloParams> yoloParams;

        IEGraph::InitParams graphParams;
        graphParams.batchSize       = FLAGS_bs;
        graphParams.maxRequests     = FLAGS_nireq;
        graphParams.collectStats    = FLAGS_show_stats;
        graphParams.reportPerf      = FLAGS_pc;
        graphParams.modelPath       = modelPath;
        graphParams.cpuExtPath      = FLAGS_l;
        graphParams.cldnnConfigPath = FLAGS_c;
        graphParams.deviceName      = FLAGS_d;
        graphParams.postLoadFunc    = [&yoloParams](const std::vector<std::string>& outputDataBlobNames,
                                                    InferenceEngine::CNNNetwork &network) {
                                                        yoloParams = GetYoloParams(outputDataBlobNames, network);
                                                    };

        std::shared_ptr<IEGraph> network(new IEGraph(graphParams));
        auto inputDims = network->getInputDims();
        if (4 != inputDims.size()) {
            throw std::runtime_error("Invalid network input dimensions");
        }

        std::vector<std::string> files;
        parseInputFilesArguments(files);

        slog::info << "\tNumber of input web cams:    " << FLAGS_nc << slog::endl;
        slog::info << "\tNumber of input video files: " << files.size() << slog::endl;
        slog::info << "\tDuplication multiplayer:     " << FLAGS_duplicate_num << slog::endl;

        const auto duplicateFactor = (1 + FLAGS_duplicate_num);
        size_t numberOfInputs = (FLAGS_nc + files.size()) * duplicateFactor;

        if (numberOfInputs == 0) {
            throw std::runtime_error("No valid inputs were supplied");
        }

        DisplayParams params = prepareDisplayParams(numberOfInputs);

        slog::info << "\tNumber of input channels:    " << numberOfInputs << slog::endl;
        if (numberOfInputs > MAX_INPUTS) {
            throw std::logic_error("Number of inputs exceed maximum value [25]");
        }

        VideoSources::InitParams vsParams;
        vsParams.queueSize            = FLAGS_n_iqs;
        vsParams.collectStats         = FLAGS_show_stats;
        vsParams.realFps              = FLAGS_real_input_fps;
        vsParams.expectedHeight = static_cast<unsigned>(inputDims[2]);
        vsParams.expectedWidth  = static_cast<unsigned>(inputDims[3]);

        VideoSources sources(vsParams);
        if (!files.empty()) {
            slog::info << "Trying to open input video ..." << slog::endl;
            for (auto& file : files) {
                try {
                    sources.openVideo(file, false, FLAGS_loop_video);
                } catch (...) {
                    slog::info << "Cannot open video [" << file << "]" << slog::endl;
                    throw;
                }
            }
        }
        if (FLAGS_nc) {
            slog::info << "Trying to connect " << FLAGS_nc << " web cams ..." << slog::endl;
            for (size_t i = 0; i < FLAGS_nc; ++i) {
                try {
                    sources.openVideo(std::to_string(i), true, false);
                } catch (...) {
                    slog::info << "Cannot open web cam [" << i << "]" << slog::endl;
                    throw;
                }
            }
        }
        sources.start();

        size_t currentFrame = 0;

        std::vector<cv::Scalar> colors;
        if (yoloParams.size() > 0)
            for (int i = 0; i < static_cast<int>(yoloParams.begin()->second.classes); ++i)
                colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));

        network->start([&](VideoFrame& img) {
            img.sourceIdx = currentFrame;
            auto camIdx = currentFrame / duplicateFactor;
            currentFrame = (currentFrame + 1) % numberOfInputs;
            return sources.getFrame(camIdx, img);
        }, [&yoloParams](InferenceEngine::InferRequest::Ptr req,
                const std::vector<std::string>& outputDataBlobNames,
                cv::Size frameSize
                ) {
            unsigned long resized_im_h = 416;
            unsigned long resized_im_w = 416;

            std::vector<DetectionObject> objects;
            // Parsing outputs
            for (auto &output_name :outputDataBlobNames) {
                ParseYOLOV3Output(req, output_name, yoloParams[output_name], resized_im_h, resized_im_w, frameSize.height, frameSize.width, FLAGS_t, objects);
            }
            // Filtering overlapping boxes and lower confidence object
            std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
            for (size_t i = 0; i < objects.size(); ++i) {
                if (objects[i].confidence == 0)
                    continue;
                for (size_t j = i + 1; j < objects.size(); ++j)
                    if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4)
                        objects[j].confidence = 0;
            }

            std::vector<Detections> detections(1);
            detections[0].set(new std::vector<DetectionObject>);

            for (auto &object : objects) {
                if (object.confidence < FLAGS_t)
                    continue;
                detections[0].get<std::vector<DetectionObject>>().push_back(object);
            }

            return detections;
        });

        network->setDetectionConfidence(static_cast<float>(FLAGS_t));

        std::atomic<float> averageFps = {0.0f};

        std::vector<std::shared_ptr<VideoFrame>> batchRes;

        std::mutex statMutex;
        std::stringstream statStream;

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC key";
        }
        std::cout << std::endl;

        cv::Size graphSize{static_cast<int>(params.windowSize.width / 4), 60};
        Presenter presenter(FLAGS_u, params.windowSize.height - graphSize.height - 10, graphSize);

        const size_t outputQueueSize = 1;
        AsyncOutput output(FLAGS_show_stats, outputQueueSize,
        [&](const std::vector<std::shared_ptr<VideoFrame>>& result) {
            std::string str;
            if (FLAGS_show_stats) {
                std::unique_lock<std::mutex> lock(statMutex);
                str = statStream.str();
            }
            displayNSources(result, averageFps, str, params, colors, presenter);
            int key = cv::waitKey(1);
            presenter.handleKey(key);

            return (key != 27);
        });

        output.start();

        using timer = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;
        timer::time_point lastTime = timer::now();
        duration samplingTimeout(FLAGS_fps_sp);

        size_t fpsCounter = 0;

        size_t perfItersCounter = 0;

        while (sources.isRunning() || network->isRunning()) {
            bool readData = true;
            while (readData) {
                auto br = network->getBatchData(params.frameSize);
                if (br.empty()) {
                    break;
                }
                for (size_t i = 0; i < br.size(); i++) {
                    auto val = static_cast<unsigned int>(br[i]->sourceIdx);
                    auto it = find_if(batchRes.begin(), batchRes.end(), [val] (const std::shared_ptr<VideoFrame>& vf) { return vf->sourceIdx == val; } );
                    if (it != batchRes.end()) {
                        if (!FLAGS_no_show) {
                            output.push(std::move(batchRes));
                        }
                        batchRes.clear();
                        readData = false;
                    }
                    batchRes.push_back(std::move(br[i]));
                }
            }
            ++fpsCounter;

            if (!output.isAlive()) {
                break;
            }

            auto currTime = timer::now();
            auto deltaTime = (currTime - lastTime);
            if (deltaTime >= samplingTimeout) {
                auto durMsec =
                        std::chrono::duration_cast<duration>(deltaTime).count();
                auto frameTime = durMsec / static_cast<float>(fpsCounter);
                fpsCounter = 0;
                lastTime = currTime;

                if (FLAGS_no_show) {
                    slog::info << "Average Throughput : " << 1000.f/frameTime << " fps" << slog::endl;
                    if (++perfItersCounter >= FLAGS_n_sp) {
                        break;
                    }
                } else {
                    averageFps = frameTime;
                }

                if (FLAGS_show_stats) {
                    auto inputStat = sources.getStats();
                    auto inferStat = network->getStats();
                    auto outputStat = output.getStats();

                    std::unique_lock<std::mutex> lock(statMutex);
                    statStream.str(std::string());
                    statStream << std::fixed << std::setprecision(1);
                    statStream << "Input reads: ";
                    for (size_t i = 0; i < inputStat.readTimes.size(); ++i) {
                        if (0 == (i % 4)) {
                            statStream << std::endl;
                        }
                        statStream << inputStat.readTimes[i] << "ms ";
                    }
                    statStream << std::endl;
                    statStream << "HW decoding latency: "
                               << inputStat.decodingLatency << "ms";
                    statStream << std::endl;
                    statStream << "Preprocess time: "
                               << inferStat.preprocessTime << "ms";
                    statStream << std::endl;
                    statStream << "Plugin latency: "
                               << inferStat.inferTime << "ms";
                    statStream << std::endl;

                    statStream << "Render time: " << outputStat.renderTime
                               << "ms" << std::endl;

                    if (FLAGS_no_show) {
                        slog::info << statStream.str() << slog::endl;
                    }
                }
            }
        }

        network.reset();

        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
