// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include "gflags/gflags.h"
#include "monitors/presenter.h"
#include "utils/common.hpp"
#include "utils_gapi/stream_source.hpp"
#include "utils/performance_metrics.hpp"
#include "utils/slog.hpp"

#include "shared_functions.hpp"
#include "nets_configuration.hpp"
#include "text_recognition.hpp"
#include "custom_kernels.hpp"
#include "custom_nets.hpp"

#include "text_detection_demo_gapi.hpp"

static
void setLabel(cv::Mat& im, const std::string& label, const cv::Point& p) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.7;
    int thickness = 1;
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    auto textPos = p;
    textPos.x = std::max(0, p.x);
    textPos.y = std::max(textSize.height, p.y);

    cv::rectangle(im, textPos + cv::Point(0, baseline),
                  textPos + cv::Point(textSize.width, -textSize.height),
                  CV_RGB(50, 205, 50), cv::FILLED);
    cv::putText(im, label, textPos, fontface, scale, CV_RGB(255, 255, 255), thickness, 8);
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }
    if (FLAGS_m_td.empty() && FLAGS_m_tr.empty()) {
        throw std::logic_error("Neither parameter -m_td nor -m_tr is not set");
    }
    if (!FLAGS_m_tr.empty() && FLAGS_dt.empty()) {
        throw std::logic_error("Parameter -dt is not set");
    }
    return true;
}

int clip(int x, int maxVal) {
    return std::min(std::max(x, 0), maxVal);
}

int main(int argc, char* argv[]) {
    try {
        // This demo covers certain topologies only
        // Parsing and validating input arguments
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        const bool gapiStreaming = !FLAGS_gapi_regular_mode;
        const bool tdEnabled  = !FLAGS_m_td.empty();
        const bool trEnabled  = !FLAGS_m_tr.empty();
        const bool trComposite = std::string::npos != FLAGS_m_tr.find("encoder");
        const float segmConfThreshold = static_cast<float>(FLAGS_cls_pixel_thr);
        const float linkConfThreshold = static_cast<float>(FLAGS_link_pixel_thr);
        const std::array<std::string,2> encoderOutputNames { FLAGS_out_enc_hidden_name,
                                                             FLAGS_features_name        };
        const std::array<std::string,3> decoderInputNames { FLAGS_in_dec_symbol_name,
                                                            FLAGS_in_dec_hidden_name,
                                                            FLAGS_features_name         };
        const std::array<std::string,2> decoderOutputNames { FLAGS_out_dec_hidden_name,
                                                             FLAGS_out_dec_symbol_name  };
        if (FLAGS_pad.length() != 1) {
            throw std::invalid_argument("Pad symbol should be 1 character");
        }
        const char kPadSymbol = FLAGS_pad[0];
        std::string trSymbolsSet;
        if (!tryReadVocabFile(FLAGS_m_tr_ss, trSymbolsSet)) {
            trSymbolsSet = FLAGS_m_tr_ss;
        }
        if (trSymbolsSet.find(kPadSymbol) != trSymbolsSet.npos) {
            throw std::invalid_argument("Symbols set for the Text Recongition model must not "
                                        "contain the reserved symbol " + kPadSymbol);
        }

        custom::NetsConfig config(FLAGS_m_td, FLAGS_m_tr);
        if (tdEnabled) {
            config.getTDinfo();
            config.configureTD(FLAGS_d_td, FLAGS_w_td, FLAGS_h_td);
            slog::info << "The Text Detection model " << FLAGS_m_td << " is loaded to " <<
                FLAGS_d_td << slog::endl;
        }
        if (trEnabled) {
            config.getTRinputInfo();
            if (trComposite) {
                config.getTRcompositeInfo(encoderOutputNames, decoderInputNames,
                                          decoderOutputNames, FLAGS_tr_pt_first, kPadSymbol,
                                          trSymbolsSet, FLAGS_dt);
                config.configureTRcomposite(FLAGS_d_tr);
                slog::info << "The Composite Text Recognition Encoder model " << FLAGS_m_tr
                    << " is loaded to " << FLAGS_d_tr << slog::endl;
                slog::info << "The Composite Text Recognition Decoder model "
                    << config.decoderModelPath << " is loaded to " << FLAGS_d_tr << slog::endl;
            } else {
                config.getTRoutputInfo(FLAGS_tr_o_blb_nm, FLAGS_tr_pt_first, kPadSymbol,
                                       FLAGS_start_index, trSymbolsSet);
                config.configureTR(FLAGS_d_tr);
            slog::info << "The Text Recognition model " << FLAGS_m_tr << " is loaded to " <<
                FLAGS_d_tr << slog::endl;
            }
        }

        std::vector<cv::RotatedRect> emptyBox =
            { {cv::Point2f(0.0f, 0.0f), cv::Size2f(0.0f, 0.0f), 0.0f} };
        /** ---------------- Main graph of demo ---------------- **/
        // Graph input
        cv::GMat in;
        // Graph outputs
        auto outs = gapiStreaming ? cv::GOut(cv::gapi::copy(in)) : cv::GOut();

        // Size of frame
        cv::GOpaque<cv::Size> size = cv::gapi::streaming::size(in);
        cv::GArray<cv::RotatedRect> rrs;
        if (tdEnabled) {
            // Text detection
            cv::GMat inDet, det1, det2;
            if (1 == config.tdInputChannels) {
                inDet = cv::gapi::BGR2Gray(in);
            } else {
                inDet = in;
            }
            std::tie(det1, det2) = cv::gapi::infer<nets::TextDetection>(inDet);
            // ROI for each text piece
            rrs = custom::DetectionPostProcess::on(det1, det2, size, config.tdInputSize,
                                                   segmConfThreshold, linkConfThreshold,
                                                   FLAGS_max_rect_num);
        } else {
            rrs = cv::GArray<cv::RotatedRect>(emptyBox);
        }
        cv::GArray<std::vector<cv::Point2f>> pts;
        if (trEnabled) {
            cv::GMat inRec;
            if (1 == config.trInputChannels) {
                inRec = cv::gapi::BGR2Gray(in);
            } else {
                inRec = in;
            }
            // Labels preprocessed for recognition for each ROI
            cv::GArray<cv::GMat> labels;
            std::tie(labels, pts) = custom::CropLabels::on(inRec, rrs, config.trInputDims,
                                                           FLAGS_cc);
            // Text Recognition
            cv::GArray<cv::GMat> texts;
            if (trComposite) {
                cv::GArray<cv::GMat> hiddens;
                cv::GArray<cv::GMat> features;
                std::tie(hiddens, features) =
                    cv::gapi::infer2<nets::TextRecognitionEncoding>(inRec, labels);
                texts = custom::CompositeTRDecode::on(hiddens, features,
                                                      config.decoderNumClasses,
                                                      config.decoderEndToken);
            } else {
                texts = cv::gapi::infer2<nets::TextRecognition>(inRec, labels);
            }
            outs += cv::GOut(texts);
        } else {
            pts = custom::PointsFromRRects::on(rrs, size, FLAGS_cc);
        }
        outs += cv::GOut(pts);
        // Inputs and outputs of graph
        cv::GComputation graph(cv::GIn(in), std::move(outs));
        /** ---------------- End of graph ---------------- **/

        // Getting information about frame from ImagesCapture
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe);
        const cv::Mat tmp = cap->read();
        cap.reset();
        cap = openImagesCapture(FLAGS_i, FLAGS_loop, read_type::safe);

        auto compileArgs = cv::compile_args(custom::kernels(), config.networks);
        if (trComposite) {
            compileArgs.emplace_back(CompositeDecInputDescrs{
                                        cv::GMatDesc(CV_32F, config.decoderHiddenInputDims),
                                        cv::GMatDesc(CV_32F, config.decoderFeaturesInputDims)
                                     });
        }
        cv::GStreamingCompiled pipeline;
        if (gapiStreaming) {
            pipeline = graph.compileStreaming(std::move(compileArgs));
            pipeline.setSource(cv::gin(cv::gapi::wip::make_src<custom::CommonCapSrc>(cap)));
            pipeline.start();
        }

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};

        cv::Size graphSize{static_cast<int>(tmp.cols / 4), 60};
        Presenter presenter(FLAGS_u, tmp.rows - graphSize.height - 10, graphSize);

        // Output containers for results
        cv::Mat image;
        std::vector<std::vector<cv::Point2f>> outPts;
        std::vector<cv::Mat> outTexts;
        auto getOutVector = [&](){
            auto outVector = gapiStreaming ? cv::gout(image) : cv::gout();
            if (trEnabled) {
                outVector += cv::gout(outTexts);
            }
            outVector += cv::gout(outPts);
            return outVector;
        };

        /** ---------------- The execution part ---------------- **/
        PerformanceMetrics metrics;
        std::chrono::steady_clock::time_point beginFrame = std::chrono::steady_clock::now();
        while (true) {
            if (gapiStreaming) {
                if (!pipeline.pull(getOutVector())) {
                    break;
                }
            } else {
                image = cap->read();
                if (!image.data) {
                    break;
                }
                graph.apply(cv::gin(image), getOutVector(), std::move(compileArgs));
            }
            const auto numFound = outPts.size();
            int numRecognized = trEnabled ? 0 : numFound;
            for (size_t l = 0; l < numFound; l++) {
                std::string res = "";
                double conf = 1.0;
                if (trEnabled) {
                    const auto& text = outTexts[l];
                    if (text.size.dims() < 3 || text.size[2] != int(config.trAlphabet.length())) {
                        throw std::runtime_error("The text recognition model does not "
                                                 "correspond to alphabet.");
                    }
                    const float *outputDataPtr = text.ptr<float>();
                    std::vector<float> outputData(outputDataPtr, outputDataPtr + text.total());
                    if (FLAGS_dt == "simple") {
                        res = SimpleDecoder(outputData, config.trAlphabet, kPadSymbol, &conf,
                                            FLAGS_start_index);
                    } else if (FLAGS_dt == "ctc") {
                        if (FLAGS_b == 0) {
                            res = CTCGreedyDecoder(outputData, config.trAlphabet, kPadSymbol,
                                                   &conf);
                        } else {
                            res = CTCBeamSearchDecoder(outputData, config.trAlphabet, kPadSymbol,
                                                       &conf, FLAGS_b);
                        }
                    } else {
                        slog::err << "No decoder type or invalid decoder type (-dt) provided: " <<
                                     FLAGS_dt << slog::endl;
                        return -1;
                    }
                    if (FLAGS_lower) {
                        res = cv::toLowerCase(res);
                    }
                    res = conf >= FLAGS_thr ? res : "";
                    numRecognized += !res.empty() ? 1 : 0;
                }

                const auto& points = outPts[l];
                if (FLAGS_r) {
                    for (size_t i = 0; i < points.size(); i++) {
                        slog::debug << clip(static_cast<int>(points[i].x), image.cols - 1) << "," <<
                                       clip(static_cast<int>(points[i].y), image.rows - 1);
                        if (i != points.size() - 1) {
                            slog::debug << ",";
                        }
                    }
                    if (trEnabled) {
                        slog::debug << "," << res;
                    }
                    if (!points.empty()) {
                        slog::debug << slog::endl;
                    }
                }

                // Displaying the results
                if (!FLAGS_no_show && (!res.empty() || !trEnabled || FLAGS_cc)) {
                    for (size_t i = 0; i < points.size() ; i++) {
                        cv::line(image, points[i], points[(i + 1) % points.size()],
                                 cv::Scalar(50, 205, 50), 2);
                    }
                    if (!points.empty() && !res.empty()) {
                        setLabel(image, res, points[custom::getTopLeftPointIdx(points)]);
                    }
                }
            }

            // Displaying system parameters and FPS
            putHighlightedText(image, "Found: " + std::to_string(numRecognized),
                               cv::Point(10, 80), cv::FONT_HERSHEY_COMPLEX, 0.65,
                               cv::Scalar(0, 0, 255), 2);

            presenter.drawGraphs(image);
            metrics.update(beginFrame, image, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);

            videoWriter.write(image);

            if (!FLAGS_no_show) {
                cv::imshow("Press ESC or Q to exit", image);
                int key = cv::waitKey(1);
                if ('q' == key || 'Q' == key || key == 27) {
                    break;
                }
                presenter.handleKey(key);
            }
            beginFrame = std::chrono::steady_clock::now();
        }
        // Printing logs
        slog::info << slog::endl << "Metrics report:" << slog::endl;
        metrics.logTotal();
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened.\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
