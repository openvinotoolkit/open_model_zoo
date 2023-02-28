// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#include "api.hpp"
#include "models.hpp"
#include "reid_gallery.hpp"

#include <chrono>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "gflags/gflags.h"
#include "monitors/presenter.h"
#include "utils/args_helper.hpp"
#include "utils/images_capture.h"
#include "utils/ocv_common.hpp"
#include "utils/slog.hpp"
#include "utils/ocv_common.hpp"

namespace {
constexpr char h_msg[] = "show the help message and exit";
DEFINE_bool(h, false, h_msg);

constexpr char i_msg[] = "an input to process. The input must be a single image, a folder of images, video file or camera id. Default is 0";
DEFINE_string(i, "0", i_msg);

constexpr char mfd_msg[] = "path to the Face Detection model (.xml) file.";
DEFINE_string(mfd, "", mfd_msg);

constexpr char mlm_msg[] = "path to the Facial Landmarks Regression Retail model (.xml) file";
DEFINE_string(mlm, "", mlm_msg);

constexpr char mreid_msg[] = "path to the Face Recognition model (.xml) file.";
DEFINE_string(mreid, "", mreid_msg);

constexpr char mas_msg[] = "path to the Antispoofing Classification model (.xml) file.";
DEFINE_string(mas, "", mas_msg);

constexpr char tfd_msg[] = "probability threshold for face detections. Default is 0.5";
DEFINE_double(t_fd, 0.5, tfd_msg);

constexpr char input_shape_msg[] =
    "specify the input shape for detection network in (width x height) format. "
    "Input of model will be reshaped according specified shape."
    "Example: 1280x720. Shape of network input used by default.";
DEFINE_string(input_shape, "", input_shape_msg);

constexpr char exp_msg[] = "expand ratio for bbox before face recognition. Default is 1.0";
DEFINE_double(exp, 1.0, exp_msg);

constexpr char treid_msg[] = "cosine distance threshold between two vectors for face reidentification. Default is 0.7";
DEFINE_double(t_reid, 0.7, treid_msg);

constexpr char match_algo_msg[] = "(don't) use faster greedy matching algorithm in face reid.";
DEFINE_bool(greedy_reid_matching, false, match_algo_msg);

constexpr char fg_msg[] = "path to a faces gallery directory.";
DEFINE_string(fg, "", fg_msg);

constexpr char ag_msg[] = "(dont't) allow to grow faces gallery and to dump on disk.";
DEFINE_bool(allow_grow, false, ag_msg);

constexpr char cg_msg[] = "(dont't) crop images during faces gallery creation.";
DEFINE_bool(crop_gallery, false, cg_msg);

constexpr char o_msg[] = "name of the output file(s) to save.";
DEFINE_string(o, "", o_msg);

constexpr char loop_msg[] = "enable reading the input in a loop";
DEFINE_bool(loop, false, loop_msg);

constexpr char dfd_msg[] =
    "specify a device Face Detection model to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(dfd, "CPU", dfd_msg);

constexpr char dlm_msg[] =
    "specify a device for Landmarks Regression model to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(dlm, "CPU", dlm_msg);

constexpr char dreid_msg[] =
    "specify a target device for Face Reidentification model to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(dreid, "CPU", dreid_msg);

constexpr char das_msg[] =
    "specify a device for Anti-spoofing model to infer on (the list of available devices is shown below). "
    "Use '-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. "
    "Use '-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. "
    "Default is CPU";
DEFINE_string(das, "CPU", das_msg);

constexpr char lim_msg[] = "number of frames to store in output. If 0 is set, all frames are stored. Default is 1000";
DEFINE_uint32(lim, 1000, lim_msg);

constexpr char show_msg[] = "(don't) show output";
DEFINE_bool(show, true, show_msg);

constexpr char u_msg[] = "resource utilization graphs. Default is cdm. "
    "c - average CPU load, d - load distribution over cores, m - memory usage, h - hide";
DEFINE_string(u, "cdm", u_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout <<   "\t[ -h]                                                " << h_msg
                  << "\n\t[--help]                                             print help on all arguments"
                  << "\n\t[ -i <INPUT>]                                        " << i_msg
                  << "\n\t --mfd <MODEL FILE>                                  " << mfd_msg
                  << "\n\t[--mlm <MODEL FILE>]                                 " << mlm_msg
                  << "\n\t[--mreid <MODEL FILE>]                               " << mreid_msg
                  << "\n\t[--mas <MODEL FILE>]                                 " << mas_msg
                  << "\n\t[--t_fd <NUMBER>]                                    " << tfd_msg
                  << "\n\t[--input_shape <STRING>]                             " << input_shape_msg
                  << "\n\t[--t_reid <NUMBER>]                                  " << treid_msg
                  << "\n\t[--exp <NUMBER>]                                     " << exp_msg
                  << "\n\t[--greedy_reid_matching] ([--nogreedy_reid_matching])" << match_algo_msg
                  << "\n\t[--fg <GALLERY PATH>]                                " << fg_msg
                  << "\n\t[--allow_grow] ([--noallow_grow])                    " << ag_msg
                  << "\n\t[--crop_gallery] ([--nocrop_gallery])                " << cg_msg
                  << "\n\t[--dfd <DEVICE>]                                     " << dfd_msg
                  << "\n\t[--dlm <DEVICE>]                                     " << dlm_msg
                  << "\n\t[--dreid <DEVICE>]                                   " << dreid_msg
                  << "\n\t[--das <DEVICE>]                                     " << das_msg
                  << "\n\t[--lim <NUMBER>]                                     " << lim_msg
                  << "\n\t[ -o <OUTPUT>]                                       " << o_msg
                  << "\n\t[--loop]                                             " << loop_msg
                  << "\n\t[--show] ([--noshow])                                " << show_msg
                  << "\n\t[ -u <DEVICE>]                                       " << u_msg
                  << "\n\tKey bindings:"
                     "\n\t\tQ, q, Esc - Quit"
                     "\n\t\tP, p, 0, spacebar - Pause"
                     "\n\t\tC - average CPU load, D - load distribution over cores, M - memory usage, H - hide\n";
        showAvailableDevices();
        std::cout << ov::get_openvino_version() << std::endl;
        exit(0);
    } if (FLAGS_i.empty()) {
        throw std::invalid_argument{"-i <INPUT> can't be empty"};
    } if (FLAGS_mfd.empty()) {
        throw std::invalid_argument{"-m_fd <MODEL FILE> can't be empty"};
    } if (!FLAGS_fg.empty() && (FLAGS_mlm.empty() || FLAGS_mreid.empty())) {
        throw std::logic_error("Face Gallery path should be provided only with landmarks and reidentification models");
    } if (!FLAGS_input_shape.empty() && FLAGS_input_shape.find("x") == std::string::npos) {
        throw std::logic_error("Correct format of --input_shape parameter is \"width\"x\"height\".");
    }
}

cv::Size getInputSize(const std::string& input_shape) {
    size_t found = FLAGS_input_shape.find("x");
    cv::Size inputSize;
    if (found == std::string::npos) {
        inputSize = cv::Size(0, 0);
    } else {
        inputSize = cv::Size{
            std::stoi(FLAGS_input_shape.substr(0, found)),
            std::stoi(FLAGS_input_shape.substr(found + 1, FLAGS_input_shape.length()))};
    }

    return inputSize;
}

std::string getLabelForFace(const Result& result) {
    std::string faceLabel = result.label;
    if (!result.real) {
        faceLabel = "Spoof";
    }
    return faceLabel;
}

cv::Mat drawDetections(const std::vector<Result>& results, cv::Mat frame) {
    cv::Scalar acceptColor(0, 220, 0);
    cv::Scalar disableColor(0, 0, 255);
    for (const auto& result : results) {
        cv::Rect rect = result.face;
        std::string faceLabel = getLabelForFace(result);

        cv::Scalar color;

        if (result.label != EmbeddingsGallery::unknownLabel && result.real) {
            color = acceptColor;
        } else {
            color = disableColor;
        }
        int baseLine = 0;
        const cv::Size label_size = cv::getTextSize(faceLabel, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::rectangle(
            frame,
            cv::Point(rect.x, rect.y - label_size.height - baseLine),
            cv::Point(rect.x + label_size.width, rect.y),
            color, cv::FILLED);
        cv::putText(frame, faceLabel, cv::Point(rect.x, rect.y - baseLine), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::rectangle(frame, rect, color, 1);
        auto drawPhotoFrameCorner = [&](cv::Point p, int dx, int dy) {
            cv::line(frame, p, cv::Point(p.x, p.y + dy), color, 2);
            cv::line(frame, p, cv::Point(p.x + dx, p.y), color, 2);
        };

        int dx = static_cast<int>(0.1 * rect.width);
        int dy = static_cast<int>(0.1 * rect.height);

        drawPhotoFrameCorner(rect.tl(), dx, dy);
        drawPhotoFrameCorner(cv::Point(rect.x + rect.width - 1, rect.y), -dx, dy);
        drawPhotoFrameCorner(cv::Point(rect.x, rect.y + rect.height - 1), dx, -dy);
        drawPhotoFrameCorner(cv::Point(rect.x + rect.width - 1, rect.y + rect.height - 1), -dx, -dy);

        for (const auto lm : result.landmarks) {
            cv::circle(frame, lm, 2, {110, 193, 225}, -1);
        }
    }
    return frame;
}

} // namespace


int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        parse(argc, argv);

        const auto fdModelPath = FLAGS_mfd;
        const auto frModelPath = FLAGS_mreid;
        const auto lmModelPath = FLAGS_mlm;
        const auto asModelPath = FLAGS_mas;

        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        // Load face detector and create recognizer
        std::unique_ptr<FaceDetector> faceDetector;
        DetectorConfig detectorConfig(fdModelPath);
        detectorConfig.deviceName = FLAGS_dfd;
        detectorConfig.core = core;
        detectorConfig.confidenceThreshold = static_cast<float>(FLAGS_t_fd);
        detectorConfig.inputSize = getInputSize(FLAGS_input_shape);
        detectorConfig.increaseScaleX = static_cast<float>(FLAGS_exp);
        detectorConfig.increaseScaleY = static_cast<float>(FLAGS_exp);
        faceDetector.reset(new FaceDetector(detectorConfig));

        // Load lanmarks regression and reid models
        std::unique_ptr<FaceRecognizer> faceRecognizer;
        if (!lmModelPath.empty() && !frModelPath.empty()) {
            BaseConfig landmarksConfig(lmModelPath);
            landmarksConfig.deviceName = FLAGS_dlm;
            landmarksConfig.numRequests = FaceRecognizerDefault::MAX_NUM_REQUESTS;
            landmarksConfig.core = core;

            BaseConfig reidConfig(frModelPath);
            reidConfig.deviceName = FLAGS_dreid;
            reidConfig.numRequests = FaceRecognizerDefault::MAX_NUM_REQUESTS;
            reidConfig.core = core;

            bool allowGrow = FLAGS_allow_grow && FLAGS_show;

            faceRecognizer.reset(new FaceRecognizerDefault(
                landmarksConfig, reidConfig,
                detectorConfig, FLAGS_fg, FLAGS_t_reid,
                FLAGS_crop_gallery, allowGrow, FLAGS_greedy_reid_matching));
        } else {
            slog::warn << "Lanmarks Regression and Face Reidentification models are disabled!" << slog::endl;
        }

        // Load anti spoof model
        std::unique_ptr<AntiSpoofer> antiSpoofer;
        if (!asModelPath.empty()) {
            BaseConfig antiSpoofConfig(asModelPath);
            antiSpoofConfig.deviceName = FLAGS_das;
            antiSpoofConfig.numRequests = FaceRecognizerDefault::MAX_NUM_REQUESTS;
            antiSpoofConfig.core = core;
            antiSpoofer.reset(new AntiSpoofer(antiSpoofConfig));
        } else {
            slog::warn << "AntiSpoof model is disabled!" << slog::endl;
        }

        size_t framesNum = 0;

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        LazyVideoWriter videoWriter{FLAGS_o, cap->fps(), FLAGS_lim};
        cv::Mat frame = cap->read();
        cv::Size graphSize{static_cast<int>(frame.cols / 4), 60};
        Presenter presenter(FLAGS_u, frame.rows - graphSize.height - 10, graphSize);
        faceDetector->submitData(frame);
        bool keepRunning = true;
        while (keepRunning) {
            auto startTime = std::chrono::steady_clock::now();
            cv::Mat prevFrame = std::move(frame);
            frame = cap->read();

            keepRunning = !frame.empty();

            presenter.drawGraphs(prevFrame);

            std::vector<FaceBox> faces = faceDetector->getResults();

            if (keepRunning) {
                faceDetector->submitData(frame);
            }

            // Recognize
            std::vector<Result> results;
            if (faceRecognizer) {
                results = faceRecognizer->recognize(prevFrame.clone(), faces);
            } else {
                for (const auto& f : faces) {
                    results.emplace_back(f.face, std::vector<cv::Point>{}, EmbeddingsGallery::unknownId,
                        EmbeddingsGallery::unknownDistance, EmbeddingsGallery::unknownLabel);
                }
            }

            // AntiSpoof
            if (antiSpoofer) {
                antiSpoofer->process(prevFrame.clone(), faces, results);
            }

            metrics.update(startTime, frame, { 10, 22 }, cv::FONT_HERSHEY_COMPLEX, 0.65);
            prevFrame = drawDetections(results, prevFrame);

            if (FLAGS_show) {
                cv::imshow(argv[0], prevFrame);
                char key = cv::waitKey(1);
                if ('P' == key || 'p' == key || '0' == key || ' ' == key) {
                    key = cv::waitKey(0);
                }
                if (27 == key || 'q' == key || 'Q' == key) {  // Esc
                    keepRunning = false;
                }
                presenter.handleKey(key);
            }
            videoWriter.write(prevFrame);
            framesNum++;
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
