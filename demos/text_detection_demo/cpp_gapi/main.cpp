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

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include <inference_engine.hpp>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include <monitors/presenter.h>
#include <utils/common.hpp>
#include <utils_gapi/stream_source.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "shared_functions.hpp"
#include "text_recognition.hpp"
#include "custom_kernels.hpp"

#include "text_detection_demo_gapi.hpp"

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
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

void setLabel(cv::Mat& im, const std::string& label, const cv::Point& p);
int clip(int x, int maxVal) { return std::min(std::max(x, 0), maxVal); }
template<class Map> void getKeys(const Map& map, std::vector<std::string>& vec);

int main(int argc, char *argv[]) {
    try {
        /** This demo covers certain topologies only **/
        // Parsing and validating input arguments
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        PerformanceMetrics metrics;

        const auto tdModelPath = FLAGS_m_td;
        const bool tdRequired  = !tdModelPath.empty();
        const auto tdDevice    = FLAGS_d_td;

        const auto trModelPath = FLAGS_m_tr;
        const bool trRequired  = !trModelPath.empty();
        const auto trDevice    = FLAGS_d_tr;
        const auto trOutputBlobName = FLAGS_tr_o_blb_nm;
        const bool trComposite = std::string::npos != trModelPath.find("encoder");

              auto decoderType       = FLAGS_dt;
        const auto decoderBandwidth  = FLAGS_b;
        const auto decoderStartIndex = FLAGS_start_index;
        const size_t tdNewInputWidth  = FLAGS_w_td;
        const size_t tdNewInputHeight = FLAGS_h_td;
        const cv::Size tdNewInputSize = cv::Size(FLAGS_w_td, FLAGS_h_td);
        const bool tdReshape          = cv::Size() != tdNewInputSize;
        const size_t tdMaxRectsNum    = FLAGS_max_rect_num;
        const bool centralCrop        = FLAGS_cc;
        const float segmConfThreshold = static_cast<float>(FLAGS_cls_pixel_thr);
        const float linkConfThreshold = static_cast<float>(FLAGS_link_pixel_thr);
        const double trMinConfidence  = FLAGS_thr;
              bool trPadSymbolFirst   = FLAGS_tr_pt_first;

        if (FLAGS_pad.length() != 1) {
            throw std::invalid_argument("Pad symbol should be 1 character");
        }
        const char kPadSymbol = FLAGS_pad[0];

        std::string trSymbolsSet, trAlphabet;
        if (!tryReadVocabFile(FLAGS_m_tr_ss, trSymbolsSet)) {
            trSymbolsSet = FLAGS_m_tr_ss;
        }
        if (trSymbolsSet.find(kPadSymbol) != trSymbolsSet.npos) {
            throw std::invalid_argument("Symbols set for the Text Recongition model must not "
                                        "contain the reserved symbol " + kPadSymbol);
        }

        const auto inputPath = FLAGS_i;
        const bool noShow    = FLAGS_no_show;
        const bool loop      = FLAGS_loop;
        const bool rawOutput = FLAGS_r;
        const auto outputPath        = FLAGS_o;
        const uint outputFramesLimit = FLAGS_limit;

        // Getting information about frame from ImagesCapture
        std::shared_ptr<ImagesCapture> cap = openImagesCapture(inputPath, loop);
        const cv::Mat tmp = cap->read();
        cap.reset();
        if (!tmp.data) {
            throw std::runtime_error("Couldn't grab first frame");
        }
        cap = openImagesCapture(inputPath, loop);

        cv::VideoWriter videoWriter;
        if (!outputPath.empty() &&
            !videoWriter.open(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                              cap->fps(), tmp.size())) {
            throw std::runtime_error("Can't open video writer");
        }

        cv::Size graphSize{static_cast<int>(tmp.cols / 4), 60};
        Presenter presenter(FLAGS_u, tmp.rows - graphSize.height - 10, graphSize);

        std::vector<size_t>       tdInputDims{};
        cv::Size                  tdInputSize{};
        size_t                    tdInputChannels = 0;
        std::string               tdInputName = "";
        std::array<std::string,2> tdOutputNames{};
        std::vector<size_t>       trInputDims{};
        size_t                    trInputChannels = 0;
        std::array<std::string,2> trOutputNames{};
        std::string               decoderModelPath = "";
        std::array<std::string,3> decoderInputNames{};
        std::array<std::string,2> decoderOutputNames{};
        size_t                    decoderNumClasses = 0;
        size_t                    decoderEndToken = 0;
        {
            using namespace InferenceEngine;
            Core ie;
            if (tdRequired) {
                // Text Detection network
                auto network = ie.ReadNetwork(tdModelPath);
                // Getting text detection network input info
                auto inputInfo = network.getInputsInfo();
                if (1 != inputInfo.size()) {
                    throw std::runtime_error("The text detection network should have "
                                             "only one input");
                }
                tdInputName = inputInfo.begin()->first;
                tdInputDims = inputInfo.begin()->second->getInputData()->getTensorDesc().getDims();
                if (4 != tdInputDims.size()) {
                    throw std::runtime_error("The text detection network should have "
                                             "4-dimensional input");
                }
                tdInputSize = cv::Size(tdInputDims[3], tdInputDims[2]);
                tdInputChannels = tdInputDims[1];
                // Getting text detection network output names
                const size_t tdLinkLayerChannels = 16;
                const size_t tdSegmLayerChannels = 2;
                const size_t tdHorizBoxesLayerChannels  = 5;
                const size_t tdHorizLabelsLayerChannels = 0;
                auto outputInfo = network.getOutputsInfo();
                for (const auto& pair : outputInfo) {
                    switch (pair.second->getTensorDesc().getDims()[1]) {
                    case tdLinkLayerChannels:
                        tdOutputNames[0] = pair.first;
                        break;
                    case tdSegmLayerChannels:
                        tdOutputNames[1] = pair.first;
                        break;
                    case tdHorizBoxesLayerChannels:
                        tdOutputNames[0] = pair.first;
                        break;
                    case tdHorizLabelsLayerChannels:
                        tdOutputNames[1] = pair.first;
                        break;
                    default:
                        break;
                    }
                }
                if (tdOutputNames[0].empty() || tdOutputNames[1].empty()) {
                    throw std::runtime_error("Failed to determine text detection "
                                             "output layers' names");
                }
            }
            if (trRequired) {
                // Text Recognition network
                auto network = ie.ReadNetwork(trModelPath);
                // Getting text recognition network input info
                auto inputInfo = network.getInputsInfo();
                if (1 != inputInfo.size()) {
                    throw std::runtime_error("The text recognition network should have "
                                             "only one input");
                }
                trInputDims = inputInfo.begin()->second->getInputData()->
                                            getTensorDesc().getDims();
                if (4 != trInputDims.size()) {
                    throw std::runtime_error("The text recognition network should have "
                                             "4-dimensional input");
                }
                trInputChannels = trInputDims[1];
                // Getting text recognition network output info
                auto outputInfo = network.getOutputsInfo();
                if (trComposite) {
                    // This demo covers a certain composite `text-recognition-0015/0016` topology;
                    // in case of different network this might need to be changed or generalized
                    trOutputNames = { FLAGS_out_enc_hidden_name, FLAGS_features_name };

                    // Text Recognition Decoder network
                    decoderModelPath = trModelPath;
                    while (std::string::npos != decoderModelPath.find("encoder")) {
                        decoderModelPath = decoderModelPath.replace(
                            decoderModelPath.find("encoder"), 7, "decoder");
                    }
                    // Storing Decoder network input info
                    decoderInputNames =  { FLAGS_in_dec_symbol_name,
                                           FLAGS_in_dec_hidden_name,
                                           FLAGS_features_name       };
                    // Storing Decoder network output info
                    decoderOutputNames = { FLAGS_out_dec_hidden_name, FLAGS_out_dec_symbol_name };

                    // Checking names legitimacy
                    std::vector<std::string> encoderOutputLayers {};
                    getKeys(outputInfo, encoderOutputLayers);
                    auto decNetwork = ie.ReadNetwork(decoderModelPath);
                    std::vector<std::string> decoderInputLayers  {};
                    std::vector<std::string> decoderOutputLayers {};
                    getKeys(decNetwork.getInputsInfo(), decoderInputLayers);
                    getKeys(decNetwork.getOutputsInfo(), decoderOutputLayers);
                    checkCompositeNetNames(encoderOutputLayers, trOutputNames,
                                           decoderInputLayers,  decoderInputNames,
                                           decoderOutputLayers, decoderOutputNames);

                    trAlphabet = std::string(3, kPadSymbol) + trSymbolsSet;
                    decoderNumClasses = trAlphabet.length();
                    decoderEndToken = trAlphabet.find(kPadSymbol, 2);
                    if (!trPadSymbolFirst) {
                        throw std::logic_error("Flag '-tr_pt_first' was not set. "
                                               "Set the flag if you want to use composite model");
                    }
                    if ("simple" != decoderType) {
                        throw std::logic_error("Wrong decoder. "
                                               "Use --dt simple for composite model.");
                    }
                } else {
                    if ("" != trOutputBlobName) {
                        for (const auto& pair : outputInfo) {
                            if (pair.first == trOutputBlobName) {
                                trOutputNames[0] = trOutputBlobName;
                                break;
                            }
                        }
                        if (trOutputNames[0].empty()) {
                            throw std::runtime_error("The text recognition model does not have "
                                                     " output " + trOutputBlobName);
                        }
                    } else {
                        trOutputNames[0] = outputInfo.begin()->first;
                    }
                    trAlphabet = trPadSymbolFirst
                        ? std::string(decoderStartIndex + 1, kPadSymbol) + trSymbolsSet
                        : trSymbolsSet + kPadSymbol;
                }
            }
        }
        // Configuring networks
        cv::gapi::GNetPackage networks;
        cv::gapi::GNetPackage decNetwork;
        if (tdRequired) {
            auto tdNet = cv::gapi::ie::Params<nets::TextDetection> {
                tdModelPath, fileNameNoExt(tdModelPath) + ".bin", tdDevice
            }.cfgOutputLayers(tdOutputNames);
            if (tdReshape) {
                tdInputDims[2] = tdNewInputHeight;
                tdInputDims[3] = tdNewInputWidth;
                tdNet.cfgInputReshape(tdInputName, tdInputDims);
                tdInputSize = tdNewInputSize;
            }
            networks += cv::gapi::networks(tdNet);
        }
        if (trRequired) {
            if (trComposite) {
                static auto trEncNet = cv::gapi::ie::Params<nets::TextRecognitionEncoding> {
                    trModelPath, fileNameNoExt(trModelPath) + ".bin", trDevice
                }.cfgOutputLayers({trOutputNames});
                networks += cv::gapi::networks(trEncNet);

                static auto trDecNet = cv::gapi::ie::Params<nets::TextRecognitionDecoding> {
                    decoderModelPath, fileNameNoExt(decoderModelPath) + ".bin", trDevice
                }.cfgInputLayers({decoderInputNames}).cfgOutputLayers({decoderOutputNames});
                decNetwork += cv::gapi::networks(trDecNet);
            } else {
                auto trNet = cv::gapi::ie::Params<nets::TextRecognition> {
                    trModelPath, fileNameNoExt(trModelPath) + ".bin", trDevice
                }.cfgOutputLayers({trOutputNames[0]});
                networks += cv::gapi::networks(trNet);
            }
        }
        std::vector<cv::RotatedRect> emptyBox =
            { {cv::Point2f(0.0f, 0.0f), cv::Size2f(0.0f, 0.0f), 0.0f} };
        /** ---------------- Main graph of demo ---------------- **/
        // Graph input
        cv::GMat in;
        // Graph outputs
        auto outs = cv::GOut(cv::gapi::copy(in));

        // Size of frame
        cv::GOpaque<cv::Size> size = cv::gapi::streaming::size(in);
        cv::GArray<cv::RotatedRect> rrs;
        if (tdRequired) {
            // Text detection
            cv::GMat inDet, det1, det2;
            if (1 == tdInputChannels) {
                inDet = cv::gapi::BGR2Gray(in);
            } else {
                inDet = in;
            }
            std::tie(det1, det2) = cv::gapi::infer<nets::TextDetection>(inDet);
            // ROI for each text piece
            rrs = custom::DetectionPostProcess::on(det1, det2, size, tdInputSize,
                                                   segmConfThreshold, linkConfThreshold,
                                                   tdMaxRectsNum);
        } else {
            rrs = cv::GArray<cv::RotatedRect>(emptyBox);
        }
        cv::GArray<std::vector<cv::Point2f>> pts;
        if (trRequired) {
            cv::GMat inRec;
            if (1 == trInputChannels) {
                inRec = cv::gapi::BGR2Gray(in);
            } else {
                inRec = in;
            }
            // Labels preprocessed for recognition for each ROI
            cv::GArray<cv::GMat> labels;
            std::tie(labels, pts) = custom::CropLabels::on(inRec, rrs, trInputDims, centralCrop);
            // Text Recognition
            cv::GArray<cv::GMat> texts;
            if (trComposite) {
                cv::GArray<cv::GMat> hiddens;
                cv::GArray<cv::GMat> features;
                std::tie(hiddens, features) =
                    cv::gapi::infer2<nets::TextRecognitionEncoding>(inRec, labels);
                texts = custom::CompositeTRDecode::on(hiddens, features, decNetwork,
                                                      decoderNumClasses, decoderEndToken);
            } else {
                texts = cv::gapi::infer2<nets::TextRecognition>(inRec, labels);
            }
            outs += cv::GOut(texts);
        } else {
            pts = custom::PointsFromRRects::on(rrs, size, centralCrop);
        }
        outs += cv::GOut(pts);
        // Inputs and outputs of graph
        cv::GComputation graph(cv::GIn(in), std::move(outs));
        /** ---------------- End of graph ---------------- **/
        auto kernels  = custom::kernels();
        auto pipeline = graph.compileStreaming(cv::compile_args(kernels, networks));

        // Output containers for results
        cv::Mat image;
        std::vector<std::vector<cv::Point2f>> outPts;
        std::vector<cv::Mat> outTexts;
        auto outVector = cv::gout(image);
        if (trRequired) {
            outVector += cv::gout(outTexts);
        }
        outVector += cv::gout(outPts);
        /** ---------------- The execution part ---------------- **/
        pipeline.setSource(cv::gin(cv::gapi::wip::make_src<custom::CommonCapSrc>(cap)));
        pipeline.start();
        std::chrono::steady_clock::time_point beginFrame = std::chrono::steady_clock::now();
        while (pipeline.pull(std::move(outVector))) {
            const auto numFound = outPts.size();
            int numRecognized = trRequired ? 0 : numFound;
            for (std::size_t l = 0; l < numFound; l++) {
                std::string res = "";
                double conf = 1.0;
                if (trRequired) {
                    const auto &text = outTexts[l];
                    if (text.size.dims() < 3 || text.size[2] != int(trAlphabet.length())) {
                        throw std::runtime_error("The text recognition model does not "
                                                 "correspond to alphabet.");
                    }
                    const float *outputDataPtr = text.ptr<float>();
                    std::vector<float> outputData(outputDataPtr, outputDataPtr + text.total());
                    if (decoderType == "simple") {
                        res = SimpleDecoder(outputData, trAlphabet, kPadSymbol, &conf,
                                            decoderStartIndex);
                    } else if (decoderType == "ctc") {
                        if (decoderBandwidth == 0) {
                            res = CTCGreedyDecoder(outputData, trAlphabet, kPadSymbol, &conf);
                        } else {
                            res = CTCBeamSearchDecoder(outputData, trAlphabet, kPadSymbol, &conf,
                                                       decoderBandwidth);
                        }
                    } else {
                        slog::err << "No decoder type or invalid decoder type (-dt) provided: " <<
                                     decoderType << slog::endl;
                        return -1;
                    }
                    if (FLAGS_lower) {
                        res = cv::toLowerCase(res);
                    }
                    res = conf >= trMinConfidence ? res : "";
                    numRecognized += !res.empty() ? 1 : 0;
                }

                const auto &points = outPts[l];
                if (rawOutput) {
                    for (size_t i = 0; i < points.size(); i++) {
                        slog::debug << clip(static_cast<int>(points[i].x), image.cols - 1) << "," <<
                                       clip(static_cast<int>(points[i].y), image.rows - 1);
                        if (i != points.size() - 1) {
                            slog::debug << ",";
                        }
                    }
                    if (trRequired) {
                        slog::debug << "," << res;
                    }
                    if (!points.empty()) {
                        slog::debug << slog::endl;
                    }
                }

                // Displaying the results
                if (!noShow && (!res.empty() || !trRequired || centralCrop)) {
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

            if (videoWriter.isOpened() &&
                (outputFramesLimit == 0 || metrics.getFrameCount() <= outputFramesLimit)) {
                videoWriter.write(image);
            }

            if (!noShow) {
                cv::imshow("Press ESC or Q to exit", image);
                int key = cv::waitKey(1);
                if ('q' == key || 'Q' == key || key == 27) break;
                presenter.handleKey(key);
            }
            beginFrame = std::chrono::steady_clock::now();
        }
        // Printing logs
        slog::info << slog::endl << "Metrics report:" << slog::endl;
        metrics.logTotal();
    } catch (const std::exception & ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened.\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

void setLabel(cv::Mat& im, const std::string& label, const cv::Point & p) {
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

template<class Map> void getKeys(const Map& map, std::vector<std::string>& vec) {
    std::transform(map.begin(), map.end(), std::back_inserter(vec),
        [](const typename Map::value_type& pair) { return pair.first; });
}
