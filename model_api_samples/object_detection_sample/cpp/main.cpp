/*
// Copyright (C) 2018-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <stddef.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/detection_model.h>
#include <models/detection_model_centernet.h>
#include <models/detection_model_faceboxes.h>
#include <models/detection_model_retinaface.h>
#include <models/detection_model_retinaface_pt.h>
#include <models/detection_model_ssd.h>
#include <models/detection_model_yolo.h>
#include <models/detection_model_yolov3_onnx.h>
#include <models/detection_model_yolox.h>
#include <models/input_data.h>
#include <models/model_base.h>
#include <models/results.h>
#include <monitors/presenter.h>
#include <pipelines/async_pipeline.h>
#include <pipelines/metadata.h>
#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/default_flags.hpp>
#include <utils/images_capture.h>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

class ColorPalette {
private:
    std::vector<cv::Scalar> palette;

    static double getRandom(double a = 0.0, double b = 1.0) {
        static std::default_random_engine e;
        std::uniform_real_distribution<> dis(a, std::nextafter(b, std::numeric_limits<double>::max()));
        return dis(e);
    }

    static double distance(const cv::Scalar& c1, const cv::Scalar& c2) {
        auto dh = std::fmin(std::fabs(c1[0] - c2[0]), 1 - fabs(c1[0] - c2[0])) * 2;
        auto ds = std::fabs(c1[1] - c2[1]);
        auto dv = std::fabs(c1[2] - c2[2]);

        return dh * dh + ds * ds + dv * dv;
    }

    static cv::Scalar maxMinDistance(const std::vector<cv::Scalar>& colorSet,
                                     const std::vector<cv::Scalar>& colorCandidates) {
        std::vector<double> distances;
        distances.reserve(colorCandidates.size());
        for (auto& c1 : colorCandidates) {
            auto min =
                *std::min_element(colorSet.begin(), colorSet.end(), [&c1](const cv::Scalar& a, const cv::Scalar& b) {
                    return distance(c1, a) < distance(c1, b);
                });
            distances.push_back(distance(c1, min));
        }
        auto max = std::max_element(distances.begin(), distances.end());
        return colorCandidates[std::distance(distances.begin(), max)];
    }

    static cv::Scalar hsv2rgb(const cv::Scalar& hsvColor) {
        cv::Mat rgb;
        cv::Mat hsv(1, 1, CV_8UC3, hsvColor);
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
        return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
    }

public:
    explicit ColorPalette(size_t n) {
        palette.reserve(n);
        std::vector<cv::Scalar> hsvColors(1, {1., 1., 1.});
        std::vector<cv::Scalar> colorCandidates;
        size_t numCandidates = 100;

        hsvColors.reserve(n);
        colorCandidates.resize(numCandidates);
        for (size_t i = 1; i < n; ++i) {
            std::generate(colorCandidates.begin(), colorCandidates.end(), []() {
                return cv::Scalar{getRandom(), getRandom(0.8, 1.0), getRandom(0.5, 1.0)};
            });
            hsvColors.push_back(maxMinDistance(hsvColors, colorCandidates));
        }

        for (auto& hsv : hsvColors) {
            // Convert to OpenCV HSV format
            hsv[0] *= 179;
            hsv[1] *= 255;
            hsv[2] *= 255;

            palette.push_back(hsv2rgb(hsv));
        }
    }

    const cv::Scalar& operator[](size_t index) const {
        return palette[index % palette.size()];
    }
};

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(DetectionResult& result, const ColorPalette& palette) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }

    auto outputImg = result.metaData->asRef<ImageMetaData>().img;

    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    // Visualizing result data over source image
    for (auto& obj : result.objects) {
        slog::debug << " " << std::left << std::setw(9) << obj.label << " | " << std::setw(10) << obj.confidence
                    << " | " << std::setw(4) << int(obj.x) << " | " << std::setw(4) << int(obj.y) << " | "
                    << std::setw(4) << int(obj.x + obj.width) << " | " << std::setw(4) << int(obj.y + obj.height)
                    << slog::endl;
        std::ostringstream conf;
        conf << ":" << std::fixed << std::setprecision(1) << obj.confidence * 100 << '%';
        const auto& color = palette[obj.labelID];
        putHighlightedText(outputImg,
                           obj.label + conf.str(),
                           cv::Point2f(obj.x, obj.y - 5),
                           cv::FONT_HERSHEY_COMPLEX_SMALL,
                           1,
                           color,
                           2);
        cv::rectangle(outputImg, obj, color, 2);
    }

    try {
        for (auto& lmark : result.asRef<RetinaFaceDetectionResult>().landmarks) {
            cv::circle(outputImg, lmark, 2, cv::Scalar(0, 255, 255), -1);
        }
    } catch (const std::bad_cast&) {}

    return outputImg;
}

int main(int argc, char* argv[]) {
    try {
        slog::info << ov::get_openvino_version() << slog::endl;

        if (argc != 3) {
            std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_image>"
                      << std::endl;
            return EXIT_FAILURE;
        }
        // TODO: read config from IR
        // TODO: read type from IR
        std::unique_ptr<ModelBase> model;
        std::string FLAGS_m = argv[1];  // I use ssd300
        std::string FLAGS_at = "ssd";
        float FLAGS_t = 0.5f;
        std::vector<std::string> labels;
        ColorPalette palette(labels.size() > 0 ? labels.size() : 100);
        std::string FLAGS_layout;
        bool FLAGS_auto_resize = false;
        float FLAGS_iou_t = 0.5;
        std::vector<float> anchors;
        std::vector<int64_t> masks;
        if (FLAGS_at == "centernet") {
            model.reset(new ModelCenterNet(FLAGS_m, static_cast<float>(FLAGS_t), labels, FLAGS_layout));
        } else if (FLAGS_at == "faceboxes") {
            model.reset(new ModelFaceBoxes(FLAGS_m,
                                           static_cast<float>(FLAGS_t),
                                           FLAGS_auto_resize,
                                           static_cast<float>(FLAGS_iou_t),
                                           FLAGS_layout));
        } else if (FLAGS_at == "retinaface") {
            model.reset(new ModelRetinaFace(FLAGS_m,
                                            static_cast<float>(FLAGS_t),
                                            FLAGS_auto_resize,
                                            static_cast<float>(FLAGS_iou_t),
                                            FLAGS_layout));
        } else if (FLAGS_at == "retinaface-pytorch") {
            model.reset(new ModelRetinaFacePT(FLAGS_m,
                                              static_cast<float>(FLAGS_t),
                                              FLAGS_auto_resize,
                                              static_cast<float>(FLAGS_iou_t),
                                              FLAGS_layout));
        } else if (FLAGS_at == "ssd") {
            model.reset(new ModelSSD(FLAGS_m, static_cast<float>(FLAGS_t), FLAGS_auto_resize, labels, FLAGS_layout));
        } else if (FLAGS_at == "yolo") {
            bool FLAGS_yolo_af = true;  // Use advanced postprocessing/filtering algorithm for YOLO
            model.reset(new ModelYolo(FLAGS_m,
                                      static_cast<float>(FLAGS_t),
                                      FLAGS_auto_resize,
                                      FLAGS_yolo_af,
                                      static_cast<float>(FLAGS_iou_t),
                                      labels,
                                      anchors,
                                      masks,
                                      FLAGS_layout));
        } else if (FLAGS_at == "yolov3-onnx") {
            model.reset(new ModelYoloV3ONNX(FLAGS_m,
                                            static_cast<float>(FLAGS_t),
                                            labels,
                                            FLAGS_layout));
        } else if (FLAGS_at == "yolox") {
            model.reset(new ModelYoloX(FLAGS_m,
                                       static_cast<float>(FLAGS_t),
                                       static_cast<float>(FLAGS_iou_t),
                                       labels,
                                       FLAGS_layout));
        } else {
            slog::err << "No model type or invalid model type (-at) provided: " + FLAGS_at << slog::endl;
            return -1;
        }

        cv::Mat image = cv::imread(argv[2]);
        if (!image.data) {
            throw std::runtime_error{"Failed to read the image"};
        }

        ov::Core core;

        std::string device = "CPU";
        uint32_t nireq = 0, nthreads = 0;
        std::string nstreams;

        AsyncPipeline pipeline(std::move(model),
                               ConfigFactory::getUserConfig(device, nireq, nstreams, nthreads),
                               core);

        std::unique_ptr<ResultBase> result;

        pipeline.submitData(ImageInputData(image), std::make_shared<ImageMetaData>(image, std::chrono::steady_clock::now()));
        while (!result) {
            result = pipeline.getResult();
        }
        cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>(), palette);
        cv::imshow("Detection Results", outFrame);
        cv::waitKey(0);
    } catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
