// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "gflags/gflags.h"
#include "monitors/presenter.h"
#include "utils/common.hpp"
#include "utils/images_capture.h"
#include "utils/performance_metrics.hpp"
#include "utils/slog.hpp"

#include "cnn.hpp"
#include "text_detection.hpp"
#include "text_recognition.hpp"

#include "text_detection_demo.hpp"

static
std::string str_tolower(std::string& s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static
std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect& rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);

    std::vector<cv::Point2f> points;
    for (int i = 0; i < 4; i++) {
        points.emplace_back(vertices[i].x, vertices[i].y);
    }
    return points;
}

static
cv::Point topLeftPoint(const std::vector<cv::Point2f>& points, int* idx) {
    cv::Point2f most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f almost_most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    int most_left_idx = -1;
    int almost_most_left_idx = -1;

    for (size_t i = 0; i < points.size(); i++) {
        if (most_left.x > points[i].x) {
            if (most_left.x < std::numeric_limits<float>::max()) {
                almost_most_left = most_left;
                almost_most_left_idx = most_left_idx;
            }
            most_left = points[i];
            most_left_idx = static_cast<int>(i);
        }
        if (almost_most_left.x > points[i].x && points[i] != most_left) {
            almost_most_left = points[i];
            almost_most_left_idx = static_cast<int>(i);
        }
    }

    if (almost_most_left.y < most_left.y) {
        most_left = almost_most_left;
        most_left_idx = almost_most_left_idx;
    }

    *idx = most_left_idx;
    return most_left;
}

static
cv::Mat cropImage(const cv::Mat& image, const std::vector<cv::Point2f>& points, const cv::Size& target_size, int top_left_point_idx) {
    cv::Point2f point0 = points[static_cast<size_t>(top_left_point_idx)];
    cv::Point2f point1 = points[(top_left_point_idx + 1) % 4];
    cv::Point2f point2 = points[(top_left_point_idx + 2) % 4];

    cv::Mat crop(target_size, CV_8UC3, cv::Scalar(0));

    std::vector<cv::Point2f> from = { point0, point1, point2 };
    std::vector<cv::Point2f> to = {
        cv::Point2f(0.0f, 0.0f), cv::Point2f(static_cast<float>(target_size.width - 1), 0.0f),
        cv::Point2f(static_cast<float>(target_size.width - 1), static_cast<float>(target_size.height - 1))
    };

    cv::Mat M = cv::getAffineTransform(from, to);

    cv::warpAffine(image, crop, M, crop.size());

    return crop;
}

static
void setLabel(cv::Mat& im, const std::string& label, const cv::Point& p) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.7;
    int thickness = 1;
    int baseline = 0;

    cv::Size text_size = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    auto text_position = p;
    text_position.x = std::max(0, p.x);
    text_position.y = std::max(text_size.height, p.y);

    cv::rectangle(im, text_position + cv::Point(0, baseline), text_position + cv::Point(text_size.width, -text_size.height), CV_RGB(50, 205, 50), cv::FILLED);
    cv::putText(im, label, text_position, fontface, scale, CV_RGB(255, 255, 255), thickness, 8);
}

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // Parsing and validating input arguments
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

int clip(int x, int max_val) {
    return std::min(std::max(x, 0), max_val);
}

bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;
        // This demo covers one certain topology and cannot be generalized
        // Parsing and validating input arguments
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        double text_detection_postproc_time = 0;
        double text_recognition_postproc_time = 0;
        double text_crop_time = 0;

        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        std::unique_ptr<TextDetector> text_detector;
        if (!FLAGS_m_td.empty())
            text_detector = std::unique_ptr<TextDetector>(new TextDetector(FLAGS_m_td, "Text Detection", FLAGS_d_td, core,
                cv::Size(FLAGS_w_td, FLAGS_h_td), FLAGS_auto_resize));

        auto cls_conf_threshold = static_cast<float>(FLAGS_cls_pixel_thr);
        auto link_conf_threshold = static_cast<float>(FLAGS_link_pixel_thr);
        auto decoder_type = FLAGS_dt;
        auto decoder_bandwidth = FLAGS_b;
        auto decoder_start_index = FLAGS_start_index;
        auto pad = FLAGS_pad;
        auto m_tr_ss = FLAGS_m_tr_ss;

        if (pad.length() != 1)
            throw std::invalid_argument("Pad symbol should be 1 character");

        const char kPadSymbol = pad[0];

        if (fileExists(m_tr_ss)) {
            std::string symbols = "";
            std::ifstream inputFile(m_tr_ss);
            if (!inputFile.is_open())
                throw std::runtime_error("Can't open the vocab file: " + m_tr_ss);

            std::string vocabLine;
            while (std::getline(inputFile, vocabLine)) {
                if (vocabLine.length() != 1)
                    throw std::invalid_argument("Lines in the vocab file must contain 1 character");
                symbols += vocabLine;
            }

            if (symbols.empty())
                throw std::logic_error("File is empty: " + m_tr_ss);
            m_tr_ss = symbols;
        }

        if (m_tr_ss.find(kPadSymbol) != m_tr_ss.npos) {
            std::stringstream ss;
            ss << "Symbols set for the Text Recongition model must not contain the reserved symbol " << kPadSymbol;
            throw std::invalid_argument(ss.str());
        }

        std::string kAlphabet;
        std::unique_ptr<TextRecognizer> text_recognition;

        if (!FLAGS_m_tr.empty()) {
            // 2 kPadSymbol stand for START_TOKEN and PAD_TOKEN, respectively
            kAlphabet = std::string(3, kPadSymbol) + m_tr_ss;

            std::string text_recognizer_type;
            // determine by the name if it is composite model
            if (std::string(FLAGS_m_tr).find("encoder") == std::string::npos) {
                text_recognizer_type = "Monolithic Text Recognition";
                if (FLAGS_tr_pt_first)
                    kAlphabet = std::string(decoder_start_index + 1, kPadSymbol) + m_tr_ss;
                else
                    kAlphabet = m_tr_ss + kPadSymbol;
            } else {
                text_recognizer_type = "Composite Text Recognition";
                if (!FLAGS_tr_pt_first)
                    throw std::logic_error("Flag '-tr_pt_first' was not set. Set the flag if you want to use composite model");

                if (decoder_type != "simple")
                    throw std::logic_error("Wrong decoder. Use --dt simple for composite model.");
            }

            text_recognition = std::unique_ptr<TextRecognizer>(
                new TextRecognizer(
                    FLAGS_m_tr, "Composite Text Recognition", FLAGS_d_tr, core,
                    FLAGS_out_enc_hidden_name, FLAGS_out_dec_hidden_name,
                    FLAGS_in_dec_hidden_name, FLAGS_features_name, FLAGS_in_dec_symbol_name,
                    FLAGS_out_dec_symbol_name, FLAGS_tr_o_blb_nm, kAlphabet.find(kPadSymbol, 2), FLAGS_auto_resize));
        }
        const double min_text_recognition_confidence = FLAGS_thr;

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);

        auto startTime = std::chrono::steady_clock::now();
        cv::Mat image = cap->read();

        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_limit};
        cv::Size graphSize{static_cast<int>(image.cols / 4), 60};
        Presenter presenter(FLAGS_u, image.rows - graphSize.height - 10, graphSize);

        do {
            cv::Mat demoImage = image.clone();

            std::vector<cv::RotatedRect> rects;
            if (text_detector != nullptr) {
                std::map<std::string, ov::Tensor> output_tensors = text_detector->Infer(image);

                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                rects = text_detector->postProcess(output_tensors, image.size(), text_detector->input_size(),
                                    cls_conf_threshold, link_conf_threshold);
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                text_detection_postproc_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            } else {
                rects.emplace_back(cv::Point2f(0.0f, 0.0f), cv::Size2f(0.0f, 0.0f), 0.0f);
            }

            if (FLAGS_max_rect_num >= 0 && static_cast<int>(rects.size()) > FLAGS_max_rect_num) {
                std::sort(rects.begin(), rects.end(), [](const cv::RotatedRect& a, const cv::RotatedRect& b) {
                    return a.size.area() > b.size.area(); });
                rects.resize(static_cast<size_t>(FLAGS_max_rect_num));
            }

            int num_found = text_recognition != nullptr ? 0 : static_cast<int>(rects.size());

            for (const auto& rect : rects) {
                cv::Mat cropped_text;
                std::vector<cv::Point2f> points;
                int top_left_point_idx = 0;

                if (rect.size != cv::Size2f(0, 0) && text_detector != nullptr) {
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    points = floatPointsFromRotatedRect(rect);
                    topLeftPoint(points, &top_left_point_idx);
                    auto size = text_recognition == nullptr ? cv::Size(rect.size) : text_recognition->input_size();
                    cropped_text = cropImage(image, points, size, top_left_point_idx);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_crop_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                } else {
                    if (FLAGS_cc) {
                        int w = static_cast<int>(image.cols * 0.05f);
                        int h = static_cast<int>(w * 0.5f);
                        cv::Rect r(static_cast<int>(image.cols * 0.5f - w * 0.5f), static_cast<int>(image.rows * 0.5f - h * 0.5f), w, h);
                        cropped_text = image(r).clone();
                        cv::rectangle(demoImage, r, cv::Scalar(0, 0, 255), 2);
                        points.emplace_back(r.tl());
                    } else {
                        cropped_text = image;
                        points.emplace_back(0.0f, 0.0f);
                        points.emplace_back(static_cast<float>(image.cols - 1), 0.0f);
                        points.emplace_back(static_cast<float>(image.cols - 1), static_cast<float>(image.rows - 1));
                        points.emplace_back(0.0f, static_cast<float>(image.rows - 1));
                    }
                }

                std::string res = "";
                double conf = 1.0;
                if (text_recognition != nullptr) {
                    std::map<std::string, ov::Tensor> output_tensors = text_recognition->Infer(cropped_text);

                    ov::Tensor out_tensor = output_tensors.begin()->second;
                    if (FLAGS_tr_o_blb_nm != "") {
                        const auto& it = output_tensors.find(FLAGS_tr_o_blb_nm);
                        if (it == output_tensors.end()) {
                            throw std::runtime_error("The text recognition model does not have output " + FLAGS_tr_o_blb_nm);
                        }
                        out_tensor = it->second;
                    }
                    ov::Shape output_shape = out_tensor.get_shape();

                    if (output_shape.size() < 3 || output_shape[2] != kAlphabet.length()) {
                        throw std::runtime_error("The text recognition model does not correspond to alphabet.");
                    }

                    float* output_data_pointer = out_tensor.data<float>();
                    std::vector<float> output_data(output_data_pointer, output_data_pointer + output_shape[0] * output_shape[1] * output_shape[2]);

                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    if (decoder_type == "simple") {
                        res = SimpleDecoder(output_data, kAlphabet, kPadSymbol, &conf, decoder_start_index);
                    } else if (decoder_type == "ctc") {
                        if (decoder_bandwidth == 0) {
                            res = CTCGreedyDecoder(output_data, kAlphabet, kPadSymbol, &conf);
                        } else {
                            res = CTCBeamSearchDecoder(output_data, kAlphabet, kPadSymbol, &conf, decoder_bandwidth);
                        }
                    } else {
                        slog::err << "No decoder type or invalid decoder type (-dt) provided: " + decoder_type << slog::endl;
                        return -1;
                    }
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_recognition_postproc_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

                    if (FLAGS_lower)
                        res = str_tolower(res);

                    res = conf >= min_text_recognition_confidence ? res : "";
                    num_found += !res.empty() ? 1 : 0;
                }

                if (FLAGS_r) {
                    for (size_t i = 0; i < points.size(); i++) {
                        slog::debug << clip(static_cast<int>(points[i].x), image.cols - 1) << "," <<
                                       clip(static_cast<int>(points[i].y), image.rows - 1);
                        if (i != points.size() - 1)
                            slog::debug << ",";
                    }

                    if (text_recognition != nullptr) {
                        slog::debug << "," << res;
                    }

                    if (!points.empty()) {
                        slog::debug << slog::endl;
                    }
                }

                if (!FLAGS_no_show && (!res.empty() || text_recognition == nullptr)) {
                    for (size_t i = 0; i < points.size() ; i++) {
                        cv::line(demoImage, points[i], points[(i+1) % points.size()], cv::Scalar(50, 205, 50), 2);
                    }

                    if (!points.empty() && !res.empty()) {
                        setLabel(demoImage, res, points[static_cast<size_t>(top_left_point_idx)]);
                    }
                }
            }

            putHighlightedText(demoImage, "Found: " + std::to_string(num_found),
                cv::Point(10, 80), cv::FONT_HERSHEY_COMPLEX, 0.65, cv::Scalar(0, 0, 255), 2);

            presenter.drawGraphs(demoImage);
            metrics.update(startTime, demoImage, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);

            videoWriter.write(demoImage);
            if (!FLAGS_no_show) {
                cv::imshow("Press ESC or Q to exit", demoImage);
                int key = cv::waitKey(1);
                if ('q' == key || 'Q' == key || key == 27) break;
                presenter.handleKey(key);
            }

            startTime = std::chrono::steady_clock::now();
            image = cap->read();
        } while (image.data);

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();

        if (text_detector != nullptr && text_detector->ncalls()) {
            slog::info << "\tText detection inference: " << std::fixed << std::setprecision(1) << text_detector->time_elapsed() / text_detector->ncalls() << " ms" << slog::endl;
            slog::info << "\tText detection postprocessing: " << text_detection_postproc_time / text_detector->ncalls() << " ms" << slog::endl;
        }

        if (text_recognition != nullptr && text_recognition->ncalls()) {
            slog::info << "\tText recognition inference : "  << text_recognition->time_elapsed() / text_recognition->ncalls() << " ms" << slog::endl;
            slog::info << "\tText recognition postprocessing: "  << text_recognition_postproc_time / text_recognition->ncalls() / 1000 << " ms" << slog::endl;
            slog::info << "\tText crop: " << text_crop_time / text_recognition->ncalls() / 1000 << " ms" << slog::endl;
        }

        slog::info << presenter.reportMeans() << slog::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    } catch (...) {
        slog::err << "Unknown/internal exception happened.\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
