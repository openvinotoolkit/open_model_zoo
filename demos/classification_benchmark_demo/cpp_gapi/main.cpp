// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/own/assert.hpp>
#include <opencv2/gapi/streaming/source.hpp>
#include <opencv2/gapi/util/optional.hpp>
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

#include "classification_benchmark_demo_gapi.hpp"
#include "custom_kernels.hpp"
#include <models/classification_model.h>


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

}  // namespace util

namespace nets {
G_API_NET(Classification, <cv::GMat(cv::GMat)>, "classification");
}

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

        /** Get information about frame **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i,
                                                               false,
                                                               read_type::safe,
                                                               0,
                                                               std::numeric_limits<size_t>::max(),
                                                               stringToSize(FLAGS_res));
        const auto tmp = cap->read();
        cv::Size frame_size = cv::Size{tmp.cols, tmp.rows};
        cap.reset();
        // NB: oneVPL source rounds up frame size by 16
        // so size might be different from what ImagesCapture reads.
        if (FLAGS_use_onevpl) {
            frame_size.width  = cv::alignSize(frame_size.width, 16);
            frame_size.height = cv::alignSize(frame_size.height, 16);
        }

        cv::GComputation comp([&] {
            cv::GFrame in;
            cv::GOpaque<int64_t> outTs = cv::gapi::streaming::timestamp(in);

            auto size = cv::gapi::streaming::size(in);
            cv::GOpaque<cv::Rect> in_roi = custom::LocateROI::on(size);

            auto blob = cv::gapi::infer<nets::Classification>(in_roi, in);
            cv::GOpaque<IndexScore> index_score = custom::TopK::on(in, blob, FLAGS_nt);

            cv::GMat bgr = cv::gapi::streaming::BGR(in);
            auto graph_inputs = cv::GIn(in);
            return cv::GComputation(std::move(graph_inputs), cv::GOut(bgr, index_score, outTs));
        });

        /** Configure network **/
        auto config = ConfigFactory::getUserConfig(FLAGS_d, FLAGS_nireq, FLAGS_nstreams, FLAGS_nthreads);
        // clang-format off
        const auto net =
            cv::gapi::ie::Params<nets::Classification>{
                FLAGS_m,  // path to topology IR
                fileNameNoExt(FLAGS_m) + ".bin",  // path to weights
                FLAGS_d  // device specifier
            }.cfgNumRequests(config.maxAsyncRequests)
             .pluginConfig(config.getLegacyConfig());
        // clang-format on

        auto kernels = cv::gapi::combine(custom::kernels(), util::getKernelPackage(FLAGS_kernel_package));
        auto pipeline = comp.compileStreaming(cv::compile_args(kernels, cv::gapi::networks(net)));

        /** Output container for result **/
        cv::Mat output;
        IndexScore infer_result;

        /** ---------------- The execution part ---------------- **/
        FLAGS_loop = true; // override loop flag for benchmark
        cap = openImagesCapture(FLAGS_i,
                                FLAGS_loop,
                                read_type::safe,
                                0,
                                std::numeric_limits<size_t>::max(),
                                stringToSize(FLAGS_res));
        cv::gapi::wip::IStreamSource::Ptr media_cap;
        if (FLAGS_use_onevpl) {
            auto onevpl_params = util::parseVPLParams(FLAGS_onevpl_params);
            if (FLAGS_onevpl_pool_size != 0) {
                onevpl_params.push_back(
                    cv::gapi::wip::onevpl::CfgParam::create_frames_pool_size(FLAGS_onevpl_pool_size));
            }
            media_cap = cv::gapi::wip::make_onevpl_src(FLAGS_i, std::move(onevpl_params));
        } else {
            media_cap = cv::gapi::wip::make_src<custom::MediaCommonCapSrc>(cap);
        }

        auto pipeline_inputs = cv::gin(std::move(media_cap));
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
        while (keepRunning && elapsedSeconds < std::chrono::seconds(FLAGS_time) && pipeline.pull(cv::gout(output, infer_result, timestamp))) {
            std::chrono::milliseconds dur(timestamp);
            std::chrono::time_point<std::chrono::steady_clock> frame_timestamp(dur);
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
                    framesNumOnCalculationStart = 0;
                    correctPredictionsCount = 0;
                    accuracy = 0;
                    elapsedSeconds = std::chrono::steady_clock::duration(0);
                    startTime = std::chrono::steady_clock::now();
                } else {
                    presenter.handleKey(key);
                }
            }
        }

        if (!FLAGS_gt.empty()) {
            slog::info << "Accuracy (top " << FLAGS_nt << "): " << accuracy << slog::endl;
        }

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        // TODO Not all metrics are functional
        logLatencyPerStage(readerMetrics.getTotal().latency,
                           -0.0,
                           -0.0,
                           -0.0,
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
