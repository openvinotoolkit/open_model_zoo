/*
// Copyright (C) 2018-2019 Intel Corporation
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
/**
* \brief The entry point for the Inference Engine multichannel_face_detection demo application
* \file multichannel_face_detection/main.cpp
* \example multichannel_face_detection/main.cpp
*/
#include <iostream>
#include <vector>
#include <utility>

#include <mutex>
#include <queue>
#include <chrono>
#include <sstream>
#include <memory>
#include <string>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

#include <opencv2/opencv.hpp>

#include <monitors/presenter.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>
#include <utils/args_helper.hpp>

#include "input.hpp"
#include "multichannel_params.hpp"
#include "output.hpp"
#include "threading.hpp"
#include "graph.hpp"

#include "human_pose.hpp"
#include "peak.hpp"
#include "postprocessor.hpp"
#include "render_human_pose.hpp"
#include "postprocess.hpp"

namespace {

/**
* \brief This function shows a help message
*/
void showUsage() {
    std::cout << std::endl;
    std::cout << "multi_channel_human_pose_estimation_demo [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                           " << help_message << std::endl;
    std::cout << "    -i                           " << input_message << std::endl;
    std::cout << "    -loop                        " << loop_message << std::endl;
    std::cout << "    -duplicate_num               " << duplication_channel_number_message << std::endl;
    std::cout << "    -m \"<path>\"                  " << model_path_message<< std::endl;
    std::cout << "      -l \"<absolute_path>\"       " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"       " << custom_cldnn_message << std::endl;
    std::cout << "    -d \"<device>\"                " << target_device_message << std::endl;
    std::cout << "    -bs                          " << batch_size << std::endl;
    std::cout << "    -nireq                       " << num_infer_requests << std::endl;
    std::cout << "    -n_iqs                       " << input_queue_size << std::endl;
    std::cout << "    -fps_sp                      " << fps_sampling_period << std::endl;
    std::cout << "    -n_sp                        " << num_sampling_periods << std::endl;
    std::cout << "    -no_show                     " << no_show_message << std::endl;
    std::cout << "    -show_stats                  " << show_statistics << std::endl;
    std::cout << "    -real_input_fps              " << real_input_fps << std::endl;
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

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }
    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }
    if (FLAGS_duplicate_num == 0) {
        throw std::logic_error("Parameter -duplicate_num must be positive");
    }

    return true;
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
    if (gridStepX == 0 || gridStepY == 0) {
        throw std::logic_error("Can't display every input: there are too many of them");
    }
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
                     const std::string& stats,
                     DisplayParams params,
                     Presenter& presenter,
                     PerformanceMetrics& metrics) {
    cv::Mat windowImage = cv::Mat::zeros(params.windowSize, CV_8UC3);
    auto loopBody = [&](size_t i) {
        auto& elem = data[i];
        if (!elem->frame.empty()) {
            cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
            cv::Mat windowPart = windowImage(rectFrame);
            cv::resize(elem->frame, windowPart, params.frameSize);
            renderHumanPose(elem->detections.get<std::vector<HumanPose>>(), windowPart);
        }
    };

    auto drawStats = [&]() {
        if (FLAGS_show_stats && !stats.empty()) {
            static const cv::Point posPoint = cv::Point(3*DISP_WIDTH/4, 4*DISP_HEIGHT/5);
            auto pos = posPoint + cv::Point(0, 25);
            size_t currPos = 0;
            while (true) {
                auto newPos = stats.find('\n', currPos);
                putHighlightedText(windowImage, stats.substr(currPos, newPos - currPos), pos, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
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
    for (size_t i = 0; i < data.size() - 1; ++i) {
        metrics.update(data[i]->timestamp);
    }
    metrics.update(data.back()->timestamp, windowImage, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
    cv::imshow(params.name, windowImage);
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
#if USE_TBB
        TbbArenaWrapper arena;
#endif

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::string modelPath = FLAGS_m;
        std::size_t found = modelPath.find_last_of(".");
        if (found > modelPath.size()) {
            throw std::logic_error("Invalid model name: " + modelPath + ". Expected to be <model_name>.xml");
        }

        slog::info << *InferenceEngine::GetInferenceEngineVersion() << slog::endl;
        IEGraph::InitParams graphParams;
        graphParams.batchSize       = FLAGS_bs;
        graphParams.maxRequests     = FLAGS_nireq;
        graphParams.collectStats    = FLAGS_show_stats;
        graphParams.modelPath       = modelPath;
        graphParams.cpuExtPath      = FLAGS_l;
        graphParams.cldnnConfigPath = FLAGS_c;
        graphParams.deviceName      = FLAGS_d;

        std::shared_ptr<IEGraph> network(new IEGraph(graphParams));
        auto inputDims = network->getInputDims();
        if (4 != inputDims.size()) {
            throw std::runtime_error("Invalid network input dimensions");
        }

        VideoSources::InitParams vsParams;
        vsParams.inputs               = FLAGS_i;
        vsParams.loop                 = FLAGS_loop;
        vsParams.queueSize            = FLAGS_n_iqs;
        vsParams.collectStats         = FLAGS_show_stats;
        vsParams.realFps              = FLAGS_real_input_fps;
        vsParams.expectedHeight = static_cast<unsigned>(inputDims[2]);
        vsParams.expectedWidth  = static_cast<unsigned>(inputDims[3]);

        VideoSources sources(vsParams);
        DisplayParams params = prepareDisplayParams(sources.numberOfInputs() * FLAGS_duplicate_num);
        sources.start();

        size_t currentFrame = 0;

        network->start([&](VideoFrame& img) {
            img.sourceIdx = currentFrame;
            size_t camIdx = currentFrame / FLAGS_duplicate_num;
            currentFrame = (currentFrame + 1) % (sources.numberOfInputs() * FLAGS_duplicate_num);
            return sources.getFrame(camIdx, img);
        }, [](InferenceEngine::InferRequest::Ptr req, const std::vector<std::string>& outputDataBlobNames, cv::Size frameSize) {
            auto pafsBlobIt   = req->GetBlob(outputDataBlobNames[0]);
            auto pafsDesc     = pafsBlobIt->getTensorDesc();
            auto pafsWidth    = getTensorWidth(pafsDesc);
            auto pafsHeight   = getTensorHeight(pafsDesc);
            auto pafsChannels = getTensorChannels(pafsDesc);
            auto pafsBatch    = getTensorBatch(pafsDesc);

            auto heatMapsBlobIt   = req->GetBlob(outputDataBlobNames[1]);
            auto heatMapsDesc     = heatMapsBlobIt->getTensorDesc();
            auto heatMapsWidth    = getTensorWidth(heatMapsDesc);
            auto heatMapsHeight   = getTensorHeight(heatMapsDesc);
            auto heatMapsChannels = getTensorChannels(heatMapsDesc);
            std::vector<Detections> detections(pafsBatch);


            InferenceEngine::LockedMemory<const void> heatMapsBlobMapped = InferenceEngine::as<
                InferenceEngine::MemoryBlob>(heatMapsBlobIt)->rmap();
            InferenceEngine::LockedMemory<const void> pafsBlobMapped = InferenceEngine::as<
                InferenceEngine::MemoryBlob>(pafsBlobIt)->rmap();
            for (size_t i = 0; i < pafsBatch; i++) {
                std::vector<HumanPose> poses = postprocess(
                heatMapsBlobMapped.as<float*>() + i * heatMapsWidth * heatMapsHeight * heatMapsChannels,
                heatMapsWidth * heatMapsHeight,
                keypointsNumber,
                pafsBlobMapped.as<float*>() + i * pafsWidth * pafsHeight * pafsChannels,
                pafsWidth * pafsHeight,
                pafsChannels,
                heatMapsWidth, heatMapsHeight, frameSize);

                detections[i].set(new std::vector<HumanPose>(poses.size()));
                for (decltype(poses.size()) j = 0; j < poses.size(); j++) {
                    detections[i].get<std::vector<HumanPose>>()[j] = std::move(poses[j]);
                }
            }
            return detections;
        });

        std::vector<std::shared_ptr<VideoFrame>> batchRes;

        std::mutex statMutex;
        std::stringstream statStream;


        cv::Size graphSize{static_cast<int>(params.windowSize.width / 4), 60};
        Presenter presenter(FLAGS_u, params.windowSize.height - graphSize.height - 10, graphSize);
        PerformanceMetrics metrics;

        const size_t outputQueueSize = 1;
        AsyncOutput output(FLAGS_show_stats, outputQueueSize,
        [&](const std::vector<std::shared_ptr<VideoFrame>>& result) {
            std::string str;
            if (FLAGS_show_stats) {
                std::unique_lock<std::mutex> lock(statMutex);
                str = statStream.str();
            }
            displayNSources(result, str, params, presenter, metrics);
            int key = cv::waitKey(1);
            presenter.handleKey(key);

            return (key != 27);
        });

        output.start();

        using timer = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;
        timer::time_point lastTime = timer::now();
        duration samplingTimeout(FLAGS_fps_sp);

        size_t perfItersCounter = 0;

        while (sources.isRunning() || network->isRunning()) {
            bool readData = true;
            while (readData) {
                auto br = network->getBatchData(params.frameSize);
                if (br.empty()) {
                    break; // IEGraph::getBatchData had nothing to process and returned. That means it was stopped
                }
                for (size_t i = 0; i < br.size(); i++) {
                    // this approach waits for the next input image for sourceIdx. If provided a single image,
                    // it may not show results, especially if -real_input_fps is enabled
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

            if (!output.isAlive()) {
                break;
            }

            auto currTime = timer::now();
            auto deltaTime = (currTime - lastTime);
            if (deltaTime >= samplingTimeout) {
                lastTime = currTime;
                if (FLAGS_show_stats) {
                    if (++perfItersCounter >= FLAGS_n_sp) {
                        break;
                    }
                }

                if (FLAGS_show_stats) {
                    std::unique_lock<std::mutex> lock(statMutex);
                    slog::debug << "------------------- Frame # " << perfItersCounter << "------------------" << slog::endl;
                    writeStats(slog::debug, slog::endl, sources.getStats(), network->getStats(), output.getStats());
                    statStream.str(std::string());
                    writeStats(statStream, '\n', sources.getStats(), network->getStats(), output.getStats());
                }
            }
        }

        network.reset();

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
