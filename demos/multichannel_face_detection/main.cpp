/*
// Copyright (c) 2018 Intel Corporation
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
#include <condition_variable>
#include <thread>
#include <atomic>
#include <queue>
#include <chrono>
#include <sstream>
#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

#include <samples/slog.hpp>
#include "multichannel_face_detection.hpp"
#include "input.hpp"
#include "output.hpp"
#include "graph.hpp"

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
    if (FLAGS_nc*(1+FLAGS_duplicate_num) > 16 - ((FLAGS_show_stats&&!FLAGS_no_show)?1:0)) {
        throw std::logic_error("Final number of channels exceed maximum [16]");
    }
    if (FLAGS_nc == 0) {
        throw std::logic_error("Number of input cameras must be greater 0");
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
    slog::info << "\tNumber of input channels:  " << FLAGS_nc << slog::endl;
    slog::info << "\tBatch size:                " << FLAGS_bs << slog::endl;
    slog::info << "\tNumber of infer requests:  " << FLAGS_n_ir << slog::endl;

    return true;
}

namespace {

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

void displayNSources(const std::string& name,
                     const std::vector<std::shared_ptr<VideoFrame>>& data,
                     size_t count,
                     float time,
                     const std::string& stats
                   ) {
    size_t showStatsDec = (FLAGS_show_stats ? 1 : 0);
    assert(count <= (16 - showStatsDec));
    assert(data.size() <= (16 - showStatsDec));
    static const cv::Size window_size = cv::Size(DISP_WIDTH, DISP_HEIGHT);
    cv::Mat window_image = cv::Mat::zeros(window_size, CV_8UC3);

    static const cv::Point points4[] = {
        cv::Point(0, 0),
        cv::Point(0, window_size.height/2),
        cv::Point(window_size.width/2, 0),
        cv::Point(window_size.width/2, window_size.height/2),  // Reserve for show stats option
    };

    static const cv::Point points9[] = {
        cv::Point(0, 0),
        cv::Point(0, window_size.height/3),
        cv::Point(0, 2*window_size.height/3),

        cv::Point(window_size.width/3, 0),
        cv::Point(window_size.width/3, window_size.height/3),
        cv::Point(window_size.width/3, 2*window_size.height/3),

        cv::Point(2*window_size.width/3, 0),
        cv::Point(2*window_size.width/3, window_size.height/3),
        cv::Point(2*window_size.width/3, 2*window_size.height/3),  // Reserve for show stats option
    };

    static const cv::Point points16[] = {
        cv::Point(0, 0),
        cv::Point(0, window_size.height/4),
        cv::Point(0, window_size.height/2),
        cv::Point(0, window_size.height*3/4),

        cv::Point(window_size.width/4, 0),
        cv::Point(window_size.width/4, window_size.height/4),
        cv::Point(window_size.width/4, window_size.height/2),
        cv::Point(window_size.width/4, window_size.height*3/4),

        cv::Point(window_size.width/2, 0),
        cv::Point(window_size.width/2, window_size.height/4),
        cv::Point(window_size.width/2, window_size.height/2),
        cv::Point(window_size.width/2, window_size.height*3/4),

        cv::Point(window_size.width*3/4, 0),
        cv::Point(window_size.width*3/4, window_size.height/4),
        cv::Point(window_size.width*3/4, window_size.height/2),
        cv::Point(window_size.width*3/4, window_size.height*3/4),  // Reserve for show stats option
    };

    static cv::Size frame_size;
    const cv::Point* points;
    size_t lastPos;

    if (count <= (4 - showStatsDec)) {
        frame_size =  cv::Size(window_size/2);
        points = points4;
        lastPos = 3;
    } else if (count <= (9 - showStatsDec)) {
        frame_size =  cv::Size(window_size/3);
        points = points9;
        lastPos = 8;
    } else {
        frame_size =  cv::Size(window_size/4);
        points = points16;
        lastPos = 15;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        auto& elem = data[i];
        if (!elem->frame->empty()) {
            cv::Rect rect_frame1 = cv::Rect(points[i], frame_size);
            cv::Mat window_part = window_image(rect_frame1);
            cv::resize(*elem->frame.get(), window_part, frame_size);
            drawDetections(window_part, elem->detections);
        }
    }

    char str[256];
    snprintf(str, sizeof(str), "%5.2f fps", static_cast<double>(1000.0f/time));
    cv::putText(window_image, str, cv::Point(800, 100), cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 2.0,  cv::Scalar(0, 255, 0), 2);

    if (FLAGS_show_stats && !stats.empty()) {
        auto pos = points[lastPos] + cv::Point(0, 25);
        size_t curr_pos = 0;
        while (true) {
            auto new_pos = stats.find('\n', curr_pos);
            cv::putText(window_image, stats.substr(curr_pos, new_pos - curr_pos), pos, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.8,  cv::Scalar(0, 0, 255), 1);
            if (new_pos == std::string::npos) {
                break;
            }
            pos += cv::Point(0, 25);
            curr_pos = new_pos + 1;
        }
    }

    cv::imshow(name, window_image);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        slog::info << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        size_t inputPollingTimeout = 1000;  // msec
        VideoSources sources(/*async*/true, FLAGS_duplicate_num, FLAGS_show_stats,
                              FLAGS_n_iqs, inputPollingTimeout, FLAGS_real_input_fps);

        slog::info << "Trying to connect " << FLAGS_nc << " web cams ..." << slog::endl;
        for (size_t i = 0; i < FLAGS_nc; ++i) {
            if (!sources.openVideo(std::to_string(i))) {
                slog::info << "Web cam" << i << ": returned false" << slog::endl;
                return -1;
            }
        }
        sources.start();

        std::string weights_path;
        std::string model_path = FLAGS_m;
        std::size_t found = model_path.find_last_of(".");
        if (found > model_path.size()) {
            slog::info << "Invalid model name: " << model_path << slog::endl;
            slog::info << "Expected to be <model_name>.xml" << slog::endl;
            return -1;
        }
        weights_path = model_path.substr(0, found) + ".bin";
        slog::info << "Model   path: " << model_path << slog::endl;
        slog::info << "Weights path: " << weights_path << slog::endl;

        std::shared_ptr<IEGraph> network(new IEGraph(
                                             FLAGS_show_stats,
                                             FLAGS_n_ir,
                                             FLAGS_bs,
                                             model_path,
                                             weights_path,
                                             FLAGS_l,
                                             FLAGS_c,
                                             FLAGS_d,
                                             [&](VideoFrame& img) {
            return sources.getFrame(img);
        }));
        network->setDetectionConfidence(FLAGS_t);

        std::atomic<float> average_fps = {0.0f};

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
            displayNSources("Demo",
                            result, FLAGS_nc*(1 + FLAGS_duplicate_num),
                            average_fps, str);

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
            bool read_data = true;
            while (read_data) {
                auto br = network->getBatchData();
                for (size_t i = 0; i < br.size(); i++) {
                    unsigned int  val = br[i]->source_idx;
                    auto it = find_if(batchRes.begin(), batchRes.end(), [val] (const std::shared_ptr<VideoFrame>& vf) { return vf->source_idx == val; } );
                    if (it != batchRes.end()) {
                        if (!FLAGS_no_show) {
                            output.push(std::move(batchRes));
                        }
                        batchRes.clear();
                        read_data = false;
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
                    average_fps = frameTime;
                }

                if (FLAGS_show_stats) {
                    auto inputStat = sources.getStats();
                    auto inferStat = network->getStats();
                    auto outputStat = output.getStats();

                    std::unique_lock<std::mutex> lock(statMutex);
                    statStream.str(std::string());
                    statStream << std::fixed << std::setprecision(1);
                    statStream << "Input reads: ";
                    for (size_t i = 0; i < inputStat.read_times.size(); ++i) {
                        if (0 == (i % 4)) {
                            statStream << std::endl;
                        }
                        statStream << inputStat.read_times[i] << "ms ";
                    }
                    statStream << std::endl;
                    statStream << "Preprocess time: "
                               << inferStat.preprocess_time << "ms";
                    statStream << std::endl;
                    statStream << "Plugin latency: "
                               << inferStat.infer_time << "ms";
                    statStream << std::endl;

                    statStream << "Render time: " << outputStat.render_time
                               << "ms" << std::endl;

                    if (FLAGS_no_show) {
                        slog::info << statStream.str() << slog::endl;
                    }
                }
            }
        }
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
