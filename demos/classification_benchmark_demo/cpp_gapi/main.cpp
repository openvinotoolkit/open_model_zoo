// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <exception>
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
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>
#include <utils_gapi/stream_source.hpp>

#include "classification_benchmark_demo_gapi.hpp"
#include "../cpp/grid_mat.hpp"
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

    using LabelsMap = std::vector<std::pair<float, std::string>>;

    IndexScore() = default;

    void getScoredLabels(const std::vector<std::string> &in_labes,
                         LabelsMap &out_scored_labels_to_append) const {
        try {
            // fill starting from max confidence
            for (auto conf_index_it = max_confidence_with_indices.rbegin();
                 conf_index_it != max_confidence_with_indices.rend();
                 ++conf_index_it) {
                std::string str_label = in_labes.at(*conf_index_it->second);
                out_scored_labels_to_append.emplace_back(conf_index_it->first, std::move(str_label));
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

using GIndexScore = cv::GOpaque<IndexScore>;
namespace custom {
G_API_OP(TopK, <GIndexScore(cv::GFrame, cv::GMat, uint32_t)>, "classification_benchmark.custom.post_processing") {
    static cv::GOpaqueDesc outMeta(const cv::GFrameDesc &in, const cv::GMatDesc &, uint32_t) {
        return cv::empty_gopaque_desc();
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
        PerformanceMetrics metrics, renderMetrics;

        /** Get OpenVINO runtime version **/
        slog::info << ov::get_openvino_version() << slog::endl;
        // ---------- Parsing and validating of input arguments ----------
        if (!util::ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        //------------------------------ Running routines ----------------------------------------------
        std::vector<std::string> labels = ClassificationModel::loadLabels(FLAGS_labels);

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
            auto blob = cv::gapi::infer<nets::Classification>(in);
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

        auto kernels = cv::gapi::kernels<custom::OCVTopK>();
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
            throw std::runtime_error("The value of GridMat resolution flag is not valid.");
        } else {
            width = std::stoi(gridMatRowsCols[0]);
            height = std::stoi(gridMatRowsCols[1]);
        }

        GridMat gridMat(presenter, cv::Size(width, height));
        bool keepRunning = true;
        size_t framesNum = 0;
        long long correctPredictionsCount = 0;
        unsigned int framesNumOnCalculationStart = 0;
        bool isStart = true;
        double accuracy = 0;
        bool isTestMode = true;
        std::chrono::steady_clock::duration elapsedSeconds = std::chrono::steady_clock::duration(0);
        auto startTime = std::chrono::steady_clock::now();
        pipeline.start();
        IndexScore::LabelsMap top_k_scored_labels;
        top_k_scored_labels.reserve(FLAGS_nt);
        int64_t timestamp = 0;
        while (pipeline.pull(cv::gout(output, infer_result, timestamp)) && keepRunning) {
            std::chrono::milliseconds dur(timestamp);
            std::chrono::time_point<std::chrono::steady_clock> frame_timestamp(dur);
            framesNum++;

            PredictionResult predictionResult = PredictionResult::Unknown;
            std::string label("UNKNOWN");
            try {
                top_k_scored_labels.clear();
                infer_result.getScoredLabels(labels, top_k_scored_labels);

                if (top_k_scored_labels.empty()) {
                    throw std::out_of_range("No any classes detected");
                }

                label = top_k_scored_labels.begin()->second;
            } catch (const std::out_of_range &ex) {
                std::cerr << ex.what()
                          << "\nPlease, make sure the model or the label file are correct";
            }

            auto renderingStart = std::chrono::steady_clock::now();
            gridMat.updateMat(output, label, predictionResult);
            // TODO
            correctPredictionsCount = framesNum; // add ground thruth file
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
        /* TODO
         logLatencyPerStage(readerMetrics.getTotal().latency,
                           pipeline.getPreprocessMetrics().getTotal().latency,
                           pipeline.getInferenceMetircs().getTotal().latency,
                           pipeline.getPostprocessMetrics().getTotal().latency,
                           renderMetrics.getTotal().latency);
        */
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
