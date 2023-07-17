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
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gkernel.hpp>
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
#include <utils_gapi/stream_source.hpp>

#include "classification_benchmark_demo_gapi.hpp"
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

cv::gapi::wip::onevpl::CfgParam createFromString(const std::string &line) {
    using namespace cv::gapi::wip;

    if (line.empty()) {
        throw std::runtime_error("Cannot parse CfgParam from emply line");
    }

    std::string::size_type name_endline_pos = line.find(':');
    if (name_endline_pos == std::string::npos) {
        throw std::runtime_error("Cannot parse CfgParam from: " + line +
                                 "\nExpected separator \":\"");
    }

    std::string name = line.substr(0, name_endline_pos);
    std::string value = line.substr(name_endline_pos + 1);

    return cv::gapi::wip::onevpl::CfgParam::create(name, value,
                                                   /* vpp params strongly optional */
                                                   name.find("vpp.") == std::string::npos);
}

static std::vector<cv::gapi::wip::onevpl::CfgParam> parseVPLParams(const std::string& cfg_params) {
    std::vector<cv::gapi::wip::onevpl::CfgParam> source_cfgs;
    std::stringstream params_list(cfg_params);
    std::string line;
    while (std::getline(params_list, line, ',')) {
        source_cfgs.push_back(createFromString(line));
    }
    return source_cfgs;
}

static cv::gapi::GKernelPackage getKernelPackage(const std::string& type) {
    if (type == "opencv") {
        return cv::gapi::combine(cv::gapi::core::cpu::kernels(), cv::gapi::imgproc::cpu::kernels());
    } else if (type == "fluid") {
        return cv::gapi::combine(cv::gapi::core::fluid::kernels(), cv::gapi::imgproc::fluid::kernels());
    } else {
        throw std::logic_error("Unsupported kernel package type: " + type);
    }
    GAPI_Assert(false && "Unreachable code!");
}
}  // namespace util

namespace nets {
G_API_NET(Classification, <cv::GMat(cv::GMat)>, "classification");
}

struct IndexScore {
    using IndicesStorage = std::list<size_t>;
    using ConfidenceMap = std::multimap<float, typename IndicesStorage::iterator>;

    using ClassDescription = std::tuple<float, size_t, std::string>;
    using LabelsStorage = std::vector<ClassDescription>;

    IndexScore() = default;

    void getScoredLabels(const std::vector<std::string> &in_labes,
                         LabelsStorage &out_scored_labels_to_append) const {
        try {
            // fill starting from max confidence
            for (auto conf_index_it = max_confidence_with_indices.rbegin();
                 conf_index_it != max_confidence_with_indices.rend();
                 ++conf_index_it) {
                std::string str_label = in_labes.at(*conf_index_it->second);
                out_scored_labels_to_append.emplace_back(conf_index_it->first, *conf_index_it->second,  std::move(str_label));
            }
        } catch (const std::out_of_range& ex) {
            throw std::out_of_range(std::string("Provided labels file doesn't contain detected class index\nException: ") + ex.what());
        }
    }

    static IndexScore create_from_array(const float *out_blob_data_ptr, size_t out_blob_element_count,
                                        size_t top_k_amount) {
        IndexScore ret;
        if (!out_blob_data_ptr) {
            return IndexScore();
        }
        // find top K
        size_t i = 0;
        // fill & sort topK with first K elements of N array: O(K*Log(K))
        while(i < std::min(top_k_amount, out_blob_element_count)) {
            ret.max_element_indices.push_back(i);
            ret.max_confidence_with_indices.emplace(out_blob_data_ptr[i], std::prev(ret.max_element_indices.end()));
            i++;
        }

        // search K elements through remnant N-K array elements
        // greater than the minimum element in the pivot topK
        // O((N-K)*Log(K))
        for (i = top_k_amount; i < out_blob_element_count && !ret.max_confidence_with_indices.empty(); i++) {
            const auto &low_confidence_it = ret.max_confidence_with_indices.begin();
            if (out_blob_data_ptr[i] >= low_confidence_it->first) {
                auto list_min_elem_it = low_confidence_it->second;
                *list_min_elem_it = i;
                ret.max_confidence_with_indices.erase(low_confidence_it);
                ret.max_confidence_with_indices.emplace(out_blob_data_ptr[i], list_min_elem_it);
            }
        }
        return ret;
    }

private:
    ConfidenceMap max_confidence_with_indices;
    IndicesStorage max_element_indices;
};

using GRect = cv::GOpaque<cv::Rect>;
using GSize = cv::GOpaque<cv::Size>;
using GIndexScore = cv::GOpaque<IndexScore>;

namespace custom {
G_API_OP(LocateROI, <GRect(GSize)>, "sample.custom.locate-roi") {
    static cv::GOpaqueDesc outMeta(const cv::GOpaqueDesc &) {
        return cv::empty_gopaque_desc();
    }
};

G_API_OP(TopK, <GIndexScore(cv::GFrame, cv::GMat, uint32_t)>, "classification_benchmark.custom.post_processing") {
    static cv::GOpaqueDesc outMeta(const cv::GFrameDesc &in, const cv::GMatDesc &, uint32_t) {
        return cv::empty_gopaque_desc();
    }
};

GAPI_OCV_KERNEL(OCVLocateROI, LocateROI) {
    // This is the place where we can run extra analytics
    // on the input image frame and select the ROI (region
    // of interest) where we want to detect our objects (or
    // run any other inference).
    //
    // Currently it doesn't do anything intelligent,
    // but only crops the input image to square (this is
    // the most convenient aspect ratio for detectors to use)

    static void run(const cv::Size& in_size,
                    cv::Rect &out_rect) {

        // Identify the central point & square size (- some padding)
        const auto center = cv::Point{in_size.width/2, in_size.height/2};
        auto sqside = std::min(in_size.width, in_size.height);

        // Now build the central square ROI
        out_rect = cv::Rect{ center.x - sqside/2
                             , center.y - sqside/2
                             , sqside
                             , sqside
                            };
    }
};

GAPI_OCV_KERNEL(OCVTopK, TopK) {
    static void run(const cv::MediaFrame &in, const cv::Mat &out_blob, uint32_t top_k_amount, IndexScore &out) {
        // TODO extract labelId & classes

        cv::MatSize out_blob_size = out_blob.size;
        if (out_blob_size.dims() != 2) {
            throw std::runtime_error(std::string("Incorrect inference result blob dimensions has been detected: ") +
                                     std::to_string(out_blob_size.dims()) + ", expected: 2 for classification networks");
        }

        if (out_blob.type() != CV_32F) {
            throw std::runtime_error(std::string("Incorrect inference result blob elements type has been detected: ") +
                                     std::to_string(out_blob.type()) + ", expected: CV_32F for classification networks");
        }
        const float *out_blob_data_ptr = out_blob.ptr<float>();
        const size_t out_blob_data_elem_count = out_blob.total();

        out = IndexScore::create_from_array(out_blob_data_ptr, out_blob_data_elem_count, top_k_amount);
    }
};
} // namespace custom

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
        std::vector<unsigned> classIndices;
        if (!FLAGS_gt.empty()) {
            std::map<std::string, unsigned> classIndicesMap;
            std::ifstream inputGtFile(FLAGS_gt);
            if (!inputGtFile.is_open()) {
                throw std::runtime_error("Can't open the ground truth file.");
            }

            std::string line;
            while (std::getline(inputGtFile, line)) {
                size_t separatorIdx = line.find(' ');
                if (separatorIdx == std::string::npos) {
                    throw std::runtime_error("The ground truth file has incorrect format.");
                }
                std::string imagePath = line.substr(0, separatorIdx);
                size_t imagePathEndIdx = imagePath.rfind('/');
                unsigned classIndex = static_cast<unsigned>(std::stoul(line.substr(separatorIdx + 1)));
                if ((imagePathEndIdx != 1 || imagePath[0] != '.') && imagePathEndIdx != std::string::npos) {
                    throw std::runtime_error("The ground truth file has incorrect format.");
                }
                // README
                // std::map type for classIndicesMap guarantees to sort out images by name.
                // The same logic is applied in openImagesCapture() for DirReader source type,
                // which produces data for sorted pictures.
                // To be coherent in detection of ground truth for pictures we have to
                // use the same sorting approach for a source and ground truth data
                // If you're going to copy paste this code, remember that pictures need to be sorted
                classIndicesMap.insert({imagePath.substr(imagePathEndIdx + 1), classIndex});
            }

            for (size_t i = 0; i < imageNames.size(); i++) {
                auto imageSearchResult = classIndicesMap.find(imageNames[i]);
                if (imageSearchResult != classIndicesMap.end()) {
                    classIndices.push_back(imageSearchResult->second);
                } else {
                    throw std::runtime_error("No class specified for image " + imageNames[i]);
                }
            }
        } else {
            classIndices.resize(imageNames.size());
            std::fill(classIndices.begin(), classIndices.end(), 0);
        }

        //------------------------------ Running routines ----------------------------------------------
        std::vector<std::string> labels = ClassificationModel::loadLabels(FLAGS_labels);
        for (const auto& classIndex : classIndices) {
            if (classIndex >= labels.size()) {
                throw std::runtime_error("Class index " + std::to_string(classIndex) +
                                         " is outside the range supported by the model.");
            }
        }

        /** Get information about frame **/
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i,
                                                               FLAGS_loop,
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

        auto kernels = cv::gapi::kernels<custom::OCVLocateROI, custom::OCVTopK>();
        auto pipeline = comp.compileStreaming(cv::compile_args(kernels, cv::gapi::networks(net)));

        /** Output container for result **/
        cv::Mat output;
        IndexScore infer_result;

        /** ---------------- The execution part ---------------- **/
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
        bool isStart = true;
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
        while (keepRunning && elapsedSeconds < std::chrono::seconds(FLAGS_time) && pipeline.pull(cv::gout(output, infer_result, timestamp))) {
            std::chrono::milliseconds dur(timestamp);
            std::chrono::time_point<std::chrono::steady_clock> frame_timestamp(dur);
            framesNum++;

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
            try {
                top_k_scored_labels.clear();
                infer_result.getScoredLabels(labels, top_k_scored_labels);

                if (top_k_scored_labels.empty()) {
                    throw std::out_of_range("No any classes detected");
                }

                size_t predicted_class_id = std::get<1>(*top_k_scored_labels.begin());
                label = std::get<2>(*top_k_scored_labels.begin());

                if (!FLAGS_gt.empty()) {
                    if (predicted_class_id == classIndices.at(framesNum % classIndices.size())) {
                        predictionResult = PredictionResult::Correct;
                        correctPredictionsCount++;
                    }
                } else {
                    predictionResult = PredictionResult::Unknown;
                }
            } catch (const std::out_of_range &ex) {
                std::cerr << ex.what()
                          << "\nPlease, make sure the model or the label file are correct";
            }

            auto renderingStart = std::chrono::steady_clock::now();
            gridMat.updateMat(output, label, predictionResult);
            accuracy = static_cast<double>(correctPredictionsCount) / framesNum;

            gridMat.textUpdate(metrics,
                               frame_timestamp,  //from original image created timestamp
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
                int key = cv::waitKey(1);
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
