// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

#include <chrono>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/util/optional.hpp>


#include <openvino/openvino.hpp>

#include <monitors/presenter.h>
#include <utils/args_helper.hpp>
#include <utils/classification_grid_mat.hpp>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>
#include <utils_gapi/kernel_package.hpp>
#include <utils_gapi/stream_source.hpp>
#include <utils_gapi/backend_builder.hpp>

#include "classification_benchmark_demo_gapi.hpp"
#include "custom_kernels.hpp"
#include <models/classification_model.h>

namespace nets {
G_API_NET(Classification, <cv::GMat(cv::GMat)>, "classification");
}

namespace util {
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    /** ---------- Parsing and validating input arguments ----------**/
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    if (FLAGS_i.empty())
        throw std::logic_error("Parameter -i is not set");
    if (FLAGS_m.empty())
        throw std::logic_error("Parameter -m is not set");
    if (FLAGS_labels.empty()) {
        throw std::logic_error("Parameter -labels is not set");
    }
    return true;
}

inference_backends_t ParseInferenceBackends(const std::string &str, char sep = ',') {
    inference_backends_t backends;
    std::stringstream params_list(str);
    std::string line;
    while (std::getline(params_list, line, sep)) {
        backends.push(BackendDescription::parseFromArgs(line));
    }
    return backends;
}
}  // namespace util

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics, readerMetrics, renderMetrics;

        /** Get OpenVINO runtime version **/
        slog::info << ov::get_openvino_version() << slog::endl;

        // ---------- Parsing and validating of input arguments ----------
        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------- Preparing Input ------------------------------------------------------
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty())
            throw std::runtime_error("No images provided");
        std::sort(imageNames.begin(), imageNames.end());
        for (size_t i = 0; i < imageNames.size(); i++) {
            const std::string& name = imageNames[i];
            auto readingStart = std::chrono::steady_clock::now();
            const cv::Mat& tmpImage = cv::imread(name);
            if (tmpImage.data == nullptr) {
                slog::err << "Could not read image " << name << slog::endl;
                imageNames.erase(imageNames.begin() + i);
                i--;
            } else {
                readerMetrics.update(readingStart);
                size_t lastSlashIdx = name.find_last_of("/\\");
                if (lastSlashIdx != std::string::npos) {
                    imageNames[i] = name.substr(lastSlashIdx + 1);
                } else {
                    imageNames[i] = name;
                }
            }
        }

        // ----------------------------------------Read image classes-----------------------------------------
        std::vector<unsigned> classIndices = loadClassIndices(FLAGS_gt, imageNames);

        //------------------------------ Running routines ----------------------------------------------
        std::vector<std::string> labels = ClassificationModel::loadLabels(FLAGS_labels);
        for (const auto& classIndex : classIndices) {
            if (classIndex >= labels.size()) {
                throw std::runtime_error("Class index " + std::to_string(classIndex) +
                                         " is outside the range supported by the model with max index: " +
                                         std::to_string(labels.size()));
            }
        }

        cv::GComputation comp([&] {
            cv::GMat in;
            cv::GOpaque<int64_t> out_ts = cv::gapi::streaming::timestamp(in);

            auto size = cv::gapi::streaming::size(in);
            cv::GOpaque<cv::Rect> in_roi = custom::CentralCrop::on(size);

            auto blob = cv::gapi::infer<nets::Classification>(in_roi, in);
            cv::GOpaque<IndexScore> index_score = custom::TopK::on(blob, FLAGS_nt);

            auto graph_inputs = cv::GIn(in);
            return cv::GComputation(std::move(graph_inputs),
                                    cv::GOut(cv::gapi::copy(in), index_score, out_ts));
        });

        /** Configure network **/
        auto nets = cv::gapi::networks();
        auto config = ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads);
        inference_backends_t backends = util::ParseInferenceBackends(FLAGS_backend);
        std::vector<float> means;
        split(FLAGS_mean_values, ' ', means);
        std::vector<float> scales;
        split(FLAGS_scale_values, ' ', scales);
        if ((means.size() != 3 && !means.empty()) || (scales.size() != 3 && !scales.empty())) {
            throw std::runtime_error("`mean_values` and `scale_values` must be 3-components vectors "
                                     "with a space symbol as separator between component values");
        }
        nets += create_execution_network<nets::Classification>(FLAGS_m,
                                                               BackendsConfig {config,
                                                                              means,
                                                                              scales},
                                                               backends);
        auto pipeline = comp.compileStreaming(cv::compile_args(custom::kernels(),
                                              nets,
                                              cv::gapi::streaming::queue_capacity{1}));

        /** Output container for result **/
        cv::Mat output;
        IndexScore infer_result;

        /** ---------------- The execution part ---------------- **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i,
                                                               true,
                                                               read_type::safe,
                                                               0,
                                                               std::numeric_limits<size_t>::max(),
                                                               stringToSize(FLAGS_res));
        auto pipeline_inputs = cv::gin(cv::gapi::wip::make_src<custom::CommonCapSrc>(cap));
        pipeline.setSource(std::move(pipeline_inputs));
        std::string windowName = "Classification Benchmark demo G-API";
        int delay = 1;

        Presenter presenter(FLAGS_u, 0);
        int width;
        int height;
        std::vector<std::string> gridMatRowsCols = split(FLAGS_res, 'x');
        if (gridMatRowsCols.size() != 2) {
            throw std::runtime_error("The value of ClassificationGridMat resolution flag is not valid.");
        } else {
            width = std::stoi(gridMatRowsCols[0]);
            height = std::stoi(gridMatRowsCols[1]);
        }

        ClassificationGridMat gridMat(presenter, cv::Size(width, height));
        bool keepRunning = true;
        size_t framesNum = 0;
        long long correctPredictionsCount = 0;
        unsigned int framesNumOnCalculationStart = 0;
        double accuracy = 0;
        bool isTestMode = true;
        std::chrono::steady_clock::duration elapsedSeconds = std::chrono::steady_clock::duration(0);
        std::chrono::seconds testDuration = std::chrono::seconds(3);
        std::chrono::seconds fpsCalculationDuration = std::chrono::seconds(1);
        auto startTime = std::chrono::steady_clock::now();
        pipeline.start();
        IndexScore::LabelsStorage top_k_scored_labels;
        top_k_scored_labels.reserve(FLAGS_nt);
        int64_t timestamp = 0;
        size_t total_produced_image_count = 0;
        std::chrono::steady_clock::time_point output_latency_last_frame_appeared_ts = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration output_latency {0};
        while (keepRunning && elapsedSeconds < std::chrono::seconds(FLAGS_time) && pipeline.pull(cv::gout(output, infer_result, timestamp))) {
            std::chrono::microseconds dur(timestamp);
            std::chrono::time_point<std::chrono::steady_clock> frame_timestamp(dur);
            output_latency += std::chrono::steady_clock::now() - output_latency_last_frame_appeared_ts;
            framesNum++;
            size_t current_image_id = total_produced_image_count++;

            // scale ClassificationGridMat to show images bunch in 1 sec update interval
            // Logic bases on measurement frames count and time interval
            if (elapsedSeconds >= testDuration - fpsCalculationDuration && framesNumOnCalculationStart == 0) {
                framesNumOnCalculationStart = framesNum;
            }
            if (isTestMode && elapsedSeconds >= testDuration) {
                isTestMode = false;
                typedef std::chrono::duration<double, std::chrono::seconds::period> Sec;
                gridMat = ClassificationGridMat(presenter,
                                  cv::Size(width, height),
                                  cv::Size(16, 9),
                                  (framesNum - framesNumOnCalculationStart) /
                                      std::chrono::duration_cast<Sec>(fpsCalculationDuration).count());
                metrics = PerformanceMetrics();
                startTime = std::chrono::steady_clock::now();
                framesNum = 0;
                output_latency = std::chrono::steady_clock::duration{0};
                output_latency_last_frame_appeared_ts = std::chrono::steady_clock::now();
                correctPredictionsCount = 0;
                accuracy = 0;
            }

            PredictionResult predictionResult = PredictionResult::Incorrect;
            std::string label("UNKNOWN");
            top_k_scored_labels.clear();
            infer_result.getScoredLabels(labels, top_k_scored_labels);

            if (top_k_scored_labels.empty()) {
                throw std::out_of_range("No any classes detected");
            }

            // pop the most suitable label for the class index
            auto prediction_description_it = top_k_scored_labels.begin();
            label = std::get<2>(*prediction_description_it);
            if (!FLAGS_gt.empty()) {
                // iterate over topK results and compare each against ground truth
                // to extract proper classification result.
                // It may appear a class with low confidence as well
                for (; prediction_description_it != top_k_scored_labels.end();
                        ++prediction_description_it) {
                    size_t predicted_class_id_to_test = std::get<1>(*prediction_description_it);
                    if (predicted_class_id_to_test == classIndices.at(current_image_id % classIndices.size())) {
                        predictionResult = PredictionResult::Correct;
                        correctPredictionsCount++;
                        label = std::get<2>(*prediction_description_it);
                        break;
                    }
                }
            } else {
                predictionResult = PredictionResult::Unknown;
            }

            auto renderingStart = std::chrono::steady_clock::now();
            gridMat.updateMat(output, label, predictionResult);
            accuracy = static_cast<double>(correctPredictionsCount) / framesNum;

            gridMat.textUpdate(metrics,
                               frame_timestamp,
                               accuracy,
                               FLAGS_nt,
                               isTestMode,
                               !FLAGS_gt.empty(),
                               presenter);
            renderMetrics.update(renderingStart);
            elapsedSeconds = std::chrono::steady_clock::now() - startTime;
            if (!FLAGS_no_show) {
                cv::imshow("classification_demo_gapi", gridMat.outImg);
                //--- Processing keyboard events
                int key = cv::waitKey(delay);
                if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                    keepRunning = false;
                } else if (32 == key || 'r' == key ||
                            'R' == key) {  // press space or r to restart testing if needed
                    isTestMode = true;
                    framesNum = 0;
                    output_latency = std::chrono::steady_clock::duration{0};
                    output_latency_last_frame_appeared_ts = std::chrono::steady_clock::now();
                    framesNumOnCalculationStart = 0;
                    correctPredictionsCount = 0;
                    accuracy = 0;
                    elapsedSeconds = std::chrono::steady_clock::duration(0);
                    startTime = std::chrono::steady_clock::now();
                } else {
                    presenter.handleKey(key);
                }
            }
            output_latency_last_frame_appeared_ts = std::chrono::steady_clock::now();
        }

        if (!FLAGS_gt.empty()) {
            slog::info << "Accuracy (top " << FLAGS_nt << "): " << accuracy << slog::endl;
        }

        slog::info << "Metrics report:" << slog::endl;
        slog::info << "\tPipeline Latency: " << std::fixed << std::setprecision(1)
                   << metrics.getTotal().latency << " ms" << slog::endl;
        slog::info << "\tFPS: " << metrics.getTotal().fps << slog::endl;
        slog::info << "\tDecoding:\t" << std::fixed << std::setprecision(1)
                   << readerMetrics.getTotal().latency << " ms" << slog::endl;
        slog::info << "\tRendering:\t" << renderMetrics.getTotal().latency << " ms" << slog::endl;
        slog::info << "\tOutput Latency:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(output_latency).count() / framesNum
                   << " ms" << slog::endl;
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
