// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/**
* \brief The entry point for the OpenVINIO multichannel_face_detection demo application
* \file multichannel_face_detection/main.cpp
* \example multichannel_face_detection/main.cpp
*/
#include <iostream>
#include <vector>
#include <utility>

#include <algorithm>
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
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "input.hpp"
#include "multichannel_params.hpp"
#include "output.hpp"
#include "threading.hpp"
#include "graph.hpp"

namespace {
constexpr char threshold_message[] = "Probability threshold for detections";
DEFINE_double(t, 0.5, threshold_message);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    slog::info << ov::get_openvino_version() << slog::endl;
    if (FLAGS_h || argc == 1) {
        std::cout << "\n    [-h]              " << help_message
                  << "\n     -i               " << input_message
                  << "\n    [-loop]           " << loop_message
                  << "\n    [-duplicate_num]  " << duplication_channel_number_message
                  << "\n     -m <path>        " << model_path_message
                  << "\n    [-d <device>]     " << target_device_message
                  << "\n    [-bs]             " << batch_size
                  << "\n    [-n_iqs]          " << input_queue_size
                  << "\n    [-fps_sp]         " << fps_sampling_period
                  << "\n    [-n_sp]           " << num_sampling_periods
                  << "\n    [-t]              " << threshold_message
                  << "\n    [-no_show]        " << no_show_message
                  << "\n    [-show_stats]     " << show_statistics
                  << "\n    [-real_input_fps] " << real_input_fps
                  << "\n    [-u]              " << utilization_monitors_message << '\n';
        showAvailableDevices();
        std::exit(0);
    } if (FLAGS_m.empty()) {
        throw std::runtime_error("Parameter -m is not set");
    } if (FLAGS_i.empty()) {
        throw std::runtime_error("Parameter -i is not set");
    } if (FLAGS_duplicate_num == 0) {
        throw std::runtime_error("Parameter -duplicate_num must be positive");
    }
}

struct Face {
    cv::Rect2f rect;
    float confidence;
    unsigned char age;
    unsigned char gender;
    Face(cv::Rect2f r, float c, unsigned char a, unsigned char g): rect(r), confidence(c), age(a), gender(g) {}
};

void drawDetections(cv::Mat& img, const std::vector<Face>& detections) {
    for (const Face& f : detections) {
        cv::Rect ri(static_cast<int>(f.rect.x*img.cols), static_cast<int>(f.rect.y*img.rows),
                    static_cast<int>(f.rect.width*img.cols), static_cast<int>(f.rect.height*img.rows));
        cv::rectangle(img, ri, cv::Scalar(255, 0, 0), 2);
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
                     PerformanceMetrics& metrics,
                     bool no_show) {
    cv::Mat windowImage = cv::Mat::zeros(params.windowSize, CV_8UC3);
    auto loopBody = [&](size_t i) {
        auto& elem = data[i];
        if (!elem->frame.empty()) {
            cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
            cv::Mat windowPart = windowImage(rectFrame);
            cv::resize(elem->frame, windowPart, params.frameSize);
            drawDetections(windowPart, elem->detections.get<std::vector<Face>>());
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
    if (!no_show) {
        cv::imshow(params.name, windowImage);
    }
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
#if USE_TBB
        TbbArenaWrapper arena;
#endif
        parse(argc, argv);
        const std::vector<std::string>& inputs = split(FLAGS_i, ',');
        DisplayParams params = prepareDisplayParams(inputs.size() * FLAGS_duplicate_num);

        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
        if (model->get_parameters().size() != 1) {
            throw std::logic_error("Face Detection model must have only one input");
        }
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC");
        ppp.input().preprocess().convert_layout("NCHW");
        for (const ov::Output<ov::Node>& out : model->outputs()) {
            ppp.output(out.get_any_name()).tensor().set_element_type(ov::element::f32);
        }
        model = ppp.build();
        ov::set_batch(model, FLAGS_bs);
        std::queue<ov::InferRequest> reqQueue = compile(std::move(model),
            FLAGS_m, FLAGS_d, roundUp(params.count, FLAGS_bs), core);
        ov::Shape inputShape = reqQueue.front().get_input_tensor().get_shape();
        if (4 != inputShape.size()) {
            throw std::runtime_error("Invalid model input dimensions");
        }
        IEGraph graph{std::move(reqQueue), FLAGS_show_stats};

        VideoSources::InitParams vsParams;
        vsParams.inputs               = inputs;
        vsParams.loop                 = FLAGS_loop;
        vsParams.queueSize            = FLAGS_n_iqs;
        vsParams.collectStats         = FLAGS_show_stats;
        vsParams.realFps              = FLAGS_real_input_fps;
        vsParams.expectedHeight = static_cast<unsigned>(inputShape[2]);
        vsParams.expectedWidth  = static_cast<unsigned>(inputShape[3]);

        VideoSources sources(vsParams);
        sources.start();

        size_t currentFrame = 0;
        graph.start(FLAGS_bs, [&](VideoFrame& img) {
            img.sourceIdx = currentFrame;
            size_t camIdx = currentFrame / FLAGS_duplicate_num;
            currentFrame = (currentFrame + 1) % (sources.numberOfInputs() * FLAGS_duplicate_num);
            return sources.getFrame(camIdx, img);
        }, [](ov::InferRequest req, cv::Size frameSize) {
            auto output = req.get_output_tensor();
            float* dataPtr = output.data<float>();

            std::vector<Detections> detections(FLAGS_bs);
            for (auto& d : detections) {
                d.set(new std::vector<Face>);
            }

            for (size_t i = 0; i < output.get_size(); i+=7) {
                float conf = dataPtr[i + 2];
                if (conf > FLAGS_t) {
                    int idxInBatch = static_cast<int>(dataPtr[i]);
                    float x0 = std::min(std::max(0.0f, dataPtr[i + 3]), 1.0f);
                    float y0 = std::min(std::max(0.0f, dataPtr[i + 4]), 1.0f);
                    float x1 = std::min(std::max(0.0f, dataPtr[i + 5]), 1.0f);
                    float y1 = std::min(std::max(0.0f, dataPtr[i + 6]), 1.0f);

                    cv::Rect2f rect = {x0 , y0, x1-x0, y1-y0};
                    detections[idxInBatch].get<std::vector<Face>>().emplace_back(rect, conf, 0, 0);
                }
            }
            return detections;
        });

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
            displayNSources(result, str, params, presenter, metrics, FLAGS_no_show);
            int key = cv::waitKey(1);
            presenter.handleKey(key);

            return (key != 27);
        });

        output.start();

        std::vector<std::shared_ptr<VideoFrame>> batchRes;
        using timer = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;
        timer::time_point lastTime = timer::now();
        duration samplingTimeout(FLAGS_fps_sp);

        size_t perfItersCounter = 0;

        while (sources.isRunning() || graph.isRunning()) {
            bool readData = true;
            while (readData) {
                auto br = graph.getBatchData(params.frameSize);
                if (br.empty()) {
                    break;  // IEGraph::getBatchData had nothing to process and returned. That means it was stopped
                }
                for (size_t i = 0; i < br.size(); i++) {
                    // this approach waits for the next input image for sourceIdx. If provided a single image,
                    // it may not show results, especially if -real_input_fps is enabled
                    auto val = static_cast<unsigned int>(br[i]->sourceIdx);
                    auto it = find_if(batchRes.begin(), batchRes.end(), [val] (const std::shared_ptr<VideoFrame>& vf) { return vf->sourceIdx == val; } );
                    if (it != batchRes.end()) {
                        output.push(std::move(batchRes));
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
                    writeStats(slog::debug, slog::endl, sources.getStats(), graph.getStats(), output.getStats());
                    statStream.str(std::string());
                    writeStats(statStream, '\n', sources.getStats(), graph.getStats(), output.getStats());
                }
            }
        }
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
