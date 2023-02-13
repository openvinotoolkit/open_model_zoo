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

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>
#include <utils/color_palette.hpp>

// Input image is stored inside metadata, as we put it there during submission stage
cv::Mat renderDetectionData(cv::Mat& outputImg, DetectionResult& result, const DefaultColorPalette& palette) {
    if (outputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }
    // Visualizing result data over source image
    for (auto& obj : result.objects) {
        std::cout << " " << std::left << std::setw(9) << obj.label << " | " << std::setw(10) << obj.confidence
                    << " | " << std::setw(4) << int(obj.x) << " | " << std::setw(4) << int(obj.y) << " | "
                    << std::setw(4) << int(obj.x + obj.width) << " | " << std::setw(4) << int(obj.y + obj.height)
                    << std::endl;
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

    return outputImg;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << ov::get_openvino_version() << std::endl;

        if (argc != 3) {
            std::cerr << "Usage : " << argv[0] << " <path_to_model> <path_to_image>"
                      << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat image = cv::imread(argv[2]);
        if (!image.data) {
            throw std::runtime_error{"Failed to read the image"};
        }

        auto model = DetectionModel::create_model(argv[1]); // works with SSD300. Download it using Python Model API
        auto result = model->infer(ImageInputData(image));

        DefaultColorPalette palette(model->labels.size() > 0 ? model->labels.size() : 100);
        cv::Mat outFrame = renderDetectionData(image, result->asRef<DetectionResult>(), palette);

        cv::imwrite("result.png", outFrame);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return 1;
    }

    return 0;
}
