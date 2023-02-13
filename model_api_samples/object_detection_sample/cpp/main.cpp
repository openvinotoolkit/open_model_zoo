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

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(DetectionResult& result) {
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
        cv::Scalar color = {0, 255, 0};
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
        std::shared_ptr<ov::Core> core = std::make_shared<ov::Core>();
        std::unique_ptr<DetectionModel> model = DetectionModel::create_model(argv[1], core);

        cv::Mat image = cv::imread(argv[2]);
        if (!image.data) {
            throw std::runtime_error{"Failed to read the image"};
        }

        std::string device = "CPU";
        uint32_t nireq = 0, nthreads = 0;
        std::string nstreams;

        AsyncPipeline pipeline(std::move(model),
                               ConfigFactory::getUserConfig(device, nireq, nstreams, nthreads),
                               *core);

        std::unique_ptr<ResultBase> result;

        pipeline.submitData(ImageInputData(image), std::make_shared<ImageMetaData>(image, std::chrono::steady_clock::now()));
        while (!result) {
            result = pipeline.getResult();
        }
        cv::Mat outFrame = renderDetectionData(result->asRef<DetectionResult>());
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
