// Copyright (C) 2018-2019 Intel Corporation
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

#include <monitors/presenter.h>
#include <samples/common.hpp>
#include <samples/images_capture.h>
#include <samples/slog.hpp>

#include "cnn.hpp"
#include "text_detection.hpp"
#include "text_recognition.hpp"

#include "text_detection_demo.hpp"

using namespace InferenceEngine;


std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect);
std::vector<cv::Point> boundedIntPointsFromRotatedRect(const cv::RotatedRect &rect, const cv::Size& image_size);
cv::Point topLeftPoint(const std::vector<cv::Point2f> & points, int *idx);
cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size& target_size, int top_left_point_idx);
void setLabel(cv::Mat& im, const std::string& label, const cv::Point & p);

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ------------------------- Parsing and validating input arguments --------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }
    if (FLAGS_m_td.empty() && FLAGS_m_tr.empty()) {
        throw std::logic_error("Neither parameter -m_td nor -m_tr is not set");
    }
    return true;
}

int clip(int x, int max_val) {
    return std::min(std::max(x, 0), max_val);
}

int main(int argc, char *argv[]) {
    try {
        /** This demo covers one certain topology and cannot be generalized **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ----------------------------- Parsing and validating input arguments ------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        double text_detection_postproc_time = 0;
        double text_recognition_postproc_time = 0;
        double text_crop_time = 0;
        double avg_time = 0;
        const double avg_time_decay = 0.8;

        const char kPadSymbol = '#';
        if (FLAGS_m_tr_ss.find(kPadSymbol) != FLAGS_m_tr_ss.npos)
            throw std::invalid_argument("Symbols set for the Text Recongition model must not contain the reserved symbol '#'");

        std::string kAlphabet = FLAGS_m_tr_ss + kPadSymbol;

        const double min_text_recognition_confidence = FLAGS_thr;

        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        std::set<std::string> loadedDevices;
        std::vector<std::string> devices = {FLAGS_m_td.empty() ? "" : FLAGS_d_td, FLAGS_m_tr.empty() ? "" : FLAGS_d_tr};

        for (const auto &device : devices) {
            if (device.empty())
                continue;
            if (loadedDevices.find(device) != loadedDevices.end())
                continue;

            slog::info << "Device info: " << slog::endl;
            std::cout << ie.GetVersions(device) << std::endl;

            /** Load extensions for the CPU device **/
            if ((device.find("CPU") != std::string::npos)) {
                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    ie.AddExtension(extension_ptr, "CPU");
                    std::cout << "CPU Extension loaded: " << FLAGS_l << std::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for GPU
                ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            }

            loadedDevices.insert(device);
        }

        auto text_detection_model_path = FLAGS_m_td;
        auto text_recognition_model_path = FLAGS_m_tr;
        auto extension_path = FLAGS_l;
        auto cls_conf_threshold = static_cast<float>(FLAGS_cls_pixel_thr);
        auto link_conf_threshold = static_cast<float>(FLAGS_link_pixel_thr);
        auto decoder_bandwidth = FLAGS_b;

        slog::info << "Loading network files" << slog::endl;
        Cnn text_detection, text_recognition;

        if (!FLAGS_m_td.empty())
            text_detection.Init(FLAGS_m_td, ie, FLAGS_d_td, cv::Size(FLAGS_w_td, FLAGS_h_td));

        if (!FLAGS_m_tr.empty())
            text_recognition.Init(FLAGS_m_tr, ie, FLAGS_d_tr);

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        cv::Mat image = cap->read();
        if (!image.data) {
            throw std::runtime_error("Can't read an image from the input");
        }

        cv::Size graphSize{static_cast<int>(image.cols / 4), 60};
        Presenter presenter(FLAGS_u, image.rows - graphSize.height - 10, graphSize);

        slog::info << "Starting inference" << slog::endl;

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC or Q";
        }
        std::cout << std::endl;

        do {
            cv::Mat demo_image = image.clone();

            std::chrono::steady_clock::time_point begin_frame = std::chrono::steady_clock::now();
            std::vector<cv::RotatedRect> rects;
            if (text_detection.is_initialized()) {
                auto blobs = text_detection.Infer(image);
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                rects = postProcess(blobs, image.size(), text_detection.input_size(),
                                    cls_conf_threshold, link_conf_threshold);
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                text_detection_postproc_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
            } else {
                rects.emplace_back(cv::Point2f(0.0f, 0.0f), cv::Size2f(0.0f, 0.0f), 0.0f);
            }

            if (FLAGS_max_rect_num >= 0 && static_cast<int>(rects.size()) > FLAGS_max_rect_num) {
                std::sort(rects.begin(), rects.end(), [](const cv::RotatedRect & a, const cv::RotatedRect & b) {
                    return a.size.area() > b.size.area();
                });
                rects.resize(static_cast<size_t>(FLAGS_max_rect_num));
            }

            int num_found = text_recognition.is_initialized() ? 0 : static_cast<int>(rects.size());

            for (const auto &rect : rects) {
                cv::Mat cropped_text;
                std::vector<cv::Point2f> points;
                int top_left_point_idx = 0;

                if (rect.size != cv::Size2f(0, 0) && text_detection.is_initialized()) {
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    points = floatPointsFromRotatedRect(rect);
                    topLeftPoint(points, &top_left_point_idx);
                    cropped_text = cropImage(image, points, text_recognition.input_size(), top_left_point_idx);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_crop_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                } else {
                    if (FLAGS_cc) {
                        int w = static_cast<int>(image.cols * 0.05);
                        int h = static_cast<int>(w * 0.5);
                        cv::Rect r(static_cast<int>(image.cols * 0.5 - w * 0.5), static_cast<int>(image.rows * 0.5 - h * 0.5), w, h);
                        cropped_text = image(r).clone();
                        cv::rectangle(demo_image, r, cv::Scalar(0, 0, 255), 2);
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
                if (text_recognition.is_initialized()) {
                    auto blobs = text_recognition.Infer(cropped_text);
                    auto output_shape = blobs.begin()->second->getTensorDesc().getDims();
                    if (output_shape[2] != kAlphabet.length()) {
                        throw std::runtime_error("The text recognition model does not correspond to alphabet.");
                    }

                    LockedMemory<const void> blobMapped = as<MemoryBlob>(blobs.begin()->second)->rmap();
                    float *output_data_pointer = blobMapped.as<float *>();
                    std::vector<float> output_data(output_data_pointer, output_data_pointer + output_shape[0] * output_shape[2]);

                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    if (decoder_bandwidth == 0) {
                        res = CTCGreedyDecoder(output_data, kAlphabet, kPadSymbol, &conf);
                    } else {
                        res = CTCBeamSearchDecoder(output_data, kAlphabet, kPadSymbol, &conf, decoder_bandwidth);
                    }
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    text_recognition_postproc_time += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

                    res = conf >= min_text_recognition_confidence ? res : "";
                    num_found += !res.empty() ? 1 : 0;
                }

                if (FLAGS_r) {
                    for (size_t i = 0; i < points.size(); i++) {
                        std::cout << clip(static_cast<int>(points[i].x), image.cols - 1) << "," <<
                                     clip(static_cast<int>(points[i].y), image.rows - 1);
                        if (i != points.size() - 1)
                            std::cout << ",";
                    }

                    if (text_recognition.is_initialized()) {
                        std::cout << "," << res;
                    }

                    if (!points.empty()) {
                        std::cout << std::endl;
                    }
                }

                if (!FLAGS_no_show && (!res.empty() || !text_recognition.is_initialized())) {
                    for (size_t i = 0; i < points.size() ; i++) {
                        cv::line(demo_image, points[i], points[(i+1) % points.size()], cv::Scalar(50, 205, 50), 2);
                    }

                    if (!points.empty() && !res.empty()) {
                        setLabel(demo_image, res, points[static_cast<size_t>(top_left_point_idx)]);
                    }
                }
            }

            std::chrono::steady_clock::time_point end_frame = std::chrono::steady_clock::now();

            if (avg_time == 0.0) {
                avg_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count());
            } else {
                auto cur_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_frame - begin_frame).count();
                avg_time = avg_time * avg_time_decay + (1.0 - avg_time_decay) * cur_time;
            }
            int fps = static_cast<int>(1000 / avg_time);
            cv::putText(demo_image, "fps: " + std::to_string(fps) + " found: " + std::to_string(num_found),
                        cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 1);

            presenter.drawGraphs(demo_image);

            if (!FLAGS_no_show) {
                cv::imshow("Press ESC or Q to exit", demo_image);
                int key = cv::waitKey(1);
                if ('q' == key || 'Q' == key || key == 27) break;
                presenter.handleKey(key);
            }

            image = cap->read();
        } while (image.data);

        if (text_detection.ncalls() && !FLAGS_r) {
          std::cout << "text detection model inference (ms) (fps): "
                    << text_detection.time_elapsed() / text_detection.ncalls() << " "
                    << text_detection.ncalls() * 1000 / text_detection.time_elapsed() << std::endl;
          if (std::fabs(text_detection_postproc_time) < std::numeric_limits<double>::epsilon()) {
              std::cout << "text detection postprocessing: took no time " << std::endl;
          } else {
            std::cout << "text detection postprocessing (ms) (fps): "
                      << text_detection_postproc_time / text_detection.ncalls() << " "
                      << text_detection.ncalls() * 1000 / text_detection_postproc_time << std::endl << std::endl;
          }
        }

        if (text_recognition.ncalls() && !FLAGS_r) {
          std::cout << "text recognition model inference (ms) (fps): "
                    << text_recognition.time_elapsed() / text_recognition.ncalls() << " "
                    << text_recognition.ncalls() * 1000 / text_recognition.time_elapsed() << std::endl;
          if (std::fabs(text_recognition_postproc_time) < std::numeric_limits<double>::epsilon()) {
              throw std::logic_error("text_recognition_postproc_time can't be equal to zero");
          }
          std::cout << "text recognition postprocessing (ms) (fps): "
                    << text_recognition_postproc_time / text_recognition.ncalls() / 1000 << " "
                    << text_recognition.ncalls() * 1000000 / text_recognition_postproc_time << std::endl << std::endl;
          if (std::fabs(text_crop_time) < std::numeric_limits<double>::epsilon()) {
              throw std::logic_error("text_crop_time can't be equal to zero");
          }
          std::cout << "text crop (ms) (fps): " << text_crop_time / text_recognition.ncalls() / 1000 << " "
                    << text_recognition.ncalls() * 1000000 / text_crop_time << std::endl << std::endl;
        }

        // ---------------------------------------------------------------------------------------------------
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

std::vector<cv::Point2f> floatPointsFromRotatedRect(const cv::RotatedRect &rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);

    std::vector<cv::Point2f> points;
    for (int i = 0; i < 4; i++) {
        points.emplace_back(vertices[i].x, vertices[i].y);
    }
    return points;
}

cv::Point topLeftPoint(const std::vector<cv::Point2f> & points, int *idx) {
    cv::Point2f most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Point2f almost_most_left(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    int most_left_idx = -1;
    int almost_most_left_idx = -1;

    for (size_t i = 0; i < points.size() ; i++) {
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

cv::Mat cropImage(const cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Size& target_size, int top_left_point_idx) {
    cv::Point2f point0 = points[static_cast<size_t>(top_left_point_idx)];
    cv::Point2f point1 = points[(top_left_point_idx + 1) % 4];
    cv::Point2f point2 = points[(top_left_point_idx + 2) % 4];

    cv::Mat crop(target_size, CV_8UC3, cv::Scalar(0));

    std::vector<cv::Point2f> from{point0, point1, point2};
    std::vector<cv::Point2f> to{cv::Point2f(0.0f, 0.0f), cv::Point2f(static_cast<float>(target_size.width-1), 0.0f),
                                cv::Point2f(static_cast<float>(target_size.width-1), static_cast<float>(target_size.height-1))};

    cv::Mat M = cv::getAffineTransform(from, to);

    cv::warpAffine(image, crop, M, crop.size());

    return crop;
}

void setLabel(cv::Mat& im, const std::string& label, const cv::Point & p) {
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
