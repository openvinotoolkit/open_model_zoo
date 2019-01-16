// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/**
* \brief The entry point for the Inference Engine multichannel_face_detection demo application
* \file multichannel_face_detection/main.cpp
* \example multichannel_face_detection/main.cpp
*/
#include <iostream>
#include <vector>
#include <utility>

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

#include <samples/slog.hpp>

#include <samples/args_helper.hpp>

#include "input.hpp"
#include "multichannel_face_detection.hpp"
#include "output.hpp"
#include "threading.hpp"
#include "graph.hpp"

namespace {

/**
* \brief This function show a help message
*/
void showUsage() {
    std::cout << std::endl;
    std::cout << "multichannel_face_detection [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -m \"<path>\"                  " << face_detection_model_message<< std::endl;
    std::cout << "      -l \"<absolute_path>\"       " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"       " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"                " << target_device_message << std::endl;
    std::cout << "    -nc                          " << num_cameras << std::endl;
    std::cout << "    -bs                          " << batch_size << std::endl;
    std::cout << "    -n_ir                        " << num_infer_requests << std::endl;
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
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
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
    slog::info << "\tNumber of infer requests:  " << FLAGS_n_ir << slog::endl;
    slog::info << "\tNumber of input web cams:  "  << FLAGS_nc << slog::endl;

    return true;
}

void drawDetections(cv::Mat& img, const std::vector<Face>& detections) {
    for (const Face& f : detections) {
        if (f.confidence > FLAGS_t) {
            cv::Rect ri(f.rect.x*img.cols, f.rect.y*img.rows,
                        f.rect.width*img.cols, f.rect.height*img.rows);
            cv::rectangle(img, ri, cv::Scalar(255, 0, 0), 2);
        }
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

void displayNSources(const std::vector<std::shared_ptr<VideoFrame>>& data,
                     float time,
                     const std::string& stats,
                     DisplayParams params) {
    cv::Mat windowImage = cv::Mat::zeros(params.windowSize, CV_8UC3);
    auto loopBody = [&](size_t i) {
        auto& elem = data[i];
        if (!elem->frame.empty()) {
            cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
            cv::Mat windowPart = windowImage(rectFrame);
            cv::resize(elem->frame, windowPart, params.frameSize);
            drawDetections(windowPart, elem->detections);
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

        std::string weightsPath;
        std::string modelPath = FLAGS_m;
        std::size_t found = modelPath.find_last_of(".");
        if (found > modelPath.size()) {
            slog::info << "Invalid model name: " << modelPath << slog::endl;
            slog::info << "Expected to be <model_name>.xml" << slog::endl;
            return -1;
        }
        weightsPath = modelPath.substr(0, found) + ".bin";
        slog::info << "Model   path: " << modelPath << slog::endl;
        slog::info << "Weights path: " << weightsPath << slog::endl;

        IEGraph::InitParams graphParams;
        graphParams.batchSize       = FLAGS_bs;
        graphParams.maxRequests     = FLAGS_n_ir;
        graphParams.collectStats    = FLAGS_show_stats;
        graphParams.reportPerf      = FLAGS_pc;
        graphParams.modelPath       = modelPath;
        graphParams.weightsPath     = weightsPath;
        graphParams.cpuExtPath      = FLAGS_l;
        graphParams.cldnnConfigPath = FLAGS_c;
        graphParams.deviceName      = FLAGS_d;

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
                    sources.openVideo(file, false);
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
                    sources.openVideo(std::to_string(i), true);
                } catch (...) {
                    slog::info << "Cannot open web cam [" << i << "]" << slog::endl;
                    throw;
                }
            }
        }
        sources.start();

        size_t currentFrame = 0;

        network->start([&](VideoFrame& img) {
            img.sourceIdx = currentFrame;
            auto camIdx = currentFrame / duplicateFactor;
            currentFrame = (currentFrame + 1) % numberOfInputs;
            return sources.getFrame(camIdx, img);
        });

        network->setDetectionConfidence(static_cast<float>(FLAGS_t));

        std::atomic<float> averageFps = {0.0f};

        std::vector<std::shared_ptr<VideoFrame>> batchRes;

        std::mutex statMutex;
        std::stringstream statStream;

        const size_t outputQueueSize = 1;
        AsyncOutput output(FLAGS_show_stats, outputQueueSize,
        [&](const std::vector<std::shared_ptr<VideoFrame>>& result) {
            std::string str;
            if (FLAGS_show_stats) {
                std::unique_lock<std::mutex> lock(statMutex);
                str = statStream.str();
            }
            displayNSources(result, averageFps, str, params);

            return (cv::waitKey(1) != 27);
        });

        output.start();

        using timer = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;
        timer::time_point lastTime = timer::now();
        duration samplingTimeout(FLAGS_fps_sp);

        size_t fpsCounter = 0;

        size_t perfItersCounter = 0;

        while (true) {
            bool readData = true;
            while (readData) {
                auto br = network->getBatchData();
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
