// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point for inference engine Mask RCNN demo application
 * @file mask_rcnn_demo/main.cpp
 * @example mask_rcnn_demo/main.cpp
 */
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "gflags/gflags.h"
#include "utils/args_helper.hpp"
#include "utils/ocv_common.hpp"
#include "utils/performance_metrics.hpp"
#include "utils/slog.hpp"

#include "mask_rcnn_demo.h"

using namespace ov::preprocess;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // Parsing and validation of input args
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        // Parsing and validation of input args
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // This vector stores paths to the processed images
        std::vector<std::string> imagePaths;
        parseInputFilesArguments(imagePaths);
        if (imagePaths.empty())
            throw std::logic_error("No suitable images were found");

        // Init OpenVINO runtime
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::Core core;

        // Load model (Generated xml/bin files)

        // Read model
        slog::info << "Reading model: " << FLAGS_m << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
        logBasicModelInfo(model);

        // Validate inputs

        // Taking information about model inputs
        ov::OutputVector inputs = model->inputs();
        ov::OutputVector outputs = model->outputs();

        if(inputs.size() != 2 || outputs.size() != 2)
            throw std::logic_error("Expected network with 2 inputs and 2 outputs");

        size_t modelBatchSize = 0;
        size_t modelInputHeight = 0;
        size_t modelInputWidth = 0;

        ov::Layout layout;

        // Find image input dimensions
        auto it = std::find_if(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& input) {return input.get_shape().size() == 4;});
        if (it != inputs.end()) {
            // ov::set_batch() should know input layout
            layout = ov::layout::get_layout(*it);
            if (layout.empty()) {
                layout = { "NCHW" };
                model->get_parameters()[it->get_index()]->set_layout(layout);
            }

            modelBatchSize = it->get_shape()[ov::layout::batch_idx(layout)];
            modelInputHeight = it->get_shape()[ov::layout::height_idx(layout)];
            modelInputWidth = it->get_shape()[ov::layout::width_idx(layout)];
        } else {
            throw std::logic_error("Couldn't find model image input");
        }

        // Collect images
        std::vector<cv::Mat> images;

        if (modelBatchSize > imagePaths.size()) {
            slog::warn << "Model batch size is greater than number of images (" << imagePaths.size() <<
                       "), some input files will be duplicated" << slog::endl;
        } else if (modelBatchSize < imagePaths.size()) {
            modelBatchSize = imagePaths.size();
            slog::warn << "Model batch size is less than number of images (" << imagePaths.size() <<
                       "), model will be reshaped" << slog::endl;
        }

        // set batch size
        ov::set_batch(model, modelBatchSize);
        slog::info << "\tBatch size is set to " << modelBatchSize << slog::endl;

        auto startTime = std::chrono::steady_clock::now();
        for (size_t i = 0, inputIndex = 0; i < modelBatchSize; i++, inputIndex++) {
            if (inputIndex >= imagePaths.size()) {
                inputIndex = 0;
            }

            cv::Mat image = cv::imread(imagePaths[inputIndex], cv::IMREAD_COLOR);

            if (image.empty()) {
                slog::warn << "Image " + imagePaths[inputIndex] + " cannot be read!" << slog::endl;
                continue;
            }

            images.push_back(image);
        }
        if (images.empty())
            throw std::logic_error("Valid input images were not found!");

        // Load model to the device
        ov::CompiledModel compiled_model = core.compile_model(model, FLAGS_d);
        logCompiledModelInfo(compiled_model, FLAGS_m, FLAGS_d);

        // Create Infer Request
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // Set input data
        // Iterate over all the inputs
        for (size_t idx = 0; idx < inputs.size(); idx++) {
            ov::Tensor tensor = infer_request.get_input_tensor(idx);
            ov::Shape shape = tensor.get_shape();

            if (shape.size() == 4) {
                for (size_t image_id = 0; image_id < images.size(); ++image_id)
                    matToTensor(images[image_id], tensor, layout, image_id);
            }

            if (shape.size() == 2) {
                float* data = tensor.data<float>();
                data[0] = static_cast<float>(modelInputHeight); // height
                data[1] = static_cast<float>(modelInputWidth); // width
                data[2] = 1;
            }
        }

        // Do inference
        infer_request.infer();

        // Postprocess output data
        float* do_data = nullptr;
        float* masks_data = nullptr;

        size_t BOX_DESCRIPTION_SIZE = 0;

        size_t BOXES = 0;
        size_t C = 0;
        size_t H = 0;
        size_t W = 0;

        for (size_t idx = 0; idx < outputs.size(); idx++) {
            ov::Tensor tensor = infer_request.get_output_tensor(idx);
            ov::Shape shape = tensor.get_shape();
            size_t dims = shape.size();
            if (dims == 2) {
                do_data = tensor.data<float>();
                // amount of elements in each detected box description (batch, label, prob, x1, y1, x2, y2)
                BOX_DESCRIPTION_SIZE = shape[1];
            }
            if (dims == 4) {
                masks_data = tensor.data<float>();
                BOXES = shape[ov::layout::batch_idx(layout)];
                C = shape[ov::layout::channels_idx(layout)];
                H = shape[ov::layout::height_idx(layout)];
                W = shape[ov::layout::width_idx(layout)];
            }
        }

        const float PROBABILITY_THRESHOLD = 0.2f;
        // threshold used to determine whether mask pixel corresponds to object or to background
        const float MASK_THRESHOLD = 0.5f;

        size_t box_stride = W * H * C;

        std::map<size_t, size_t> class_color;

        std::vector<cv::Mat> output_images;
        for (const auto &img : images) {
            output_images.push_back(img.clone());
        }

        // Iterating over all boxes
        for (size_t box = 0; box < BOXES; ++box) {
            float* box_info = do_data + box * BOX_DESCRIPTION_SIZE;
            auto batch = static_cast<int>(box_info[0]);

            if (batch < 0)
                break;
            if (batch >= static_cast<int>(modelBatchSize))
                throw std::logic_error("Invalid batch ID within detection output box");

            float prob = box_info[2];

            float x1 = std::min(std::max(0.0f, box_info[3] * images[batch].cols), static_cast<float>(images[batch].cols));
            float y1 = std::min(std::max(0.0f, box_info[4] * images[batch].rows), static_cast<float>(images[batch].rows));
            float x2 = std::min(std::max(0.0f, box_info[5] * images[batch].cols), static_cast<float>(images[batch].cols));
            float y2 = std::min(std::max(0.0f, box_info[6] * images[batch].rows), static_cast<float>(images[batch].rows));

            int box_width = static_cast<int>(x2 - x1);
            int box_height = static_cast<int>(y2 - y1);

            auto class_id = static_cast<size_t>(box_info[1] + 1e-6);

            if (prob > PROBABILITY_THRESHOLD && box_width > 0 && box_height > 0) {
                size_t color_index = class_color.emplace(class_id, class_color.size()).first->second;
                auto& color = CITYSCAPES_COLORS[color_index % arraySize(CITYSCAPES_COLORS)];
                float* mask_arr = masks_data + box_stride * box + H * W * (class_id - 1);
                slog::info << "Detected class " << class_id << " with probability " << prob << " from batch " << batch
                           << ": [" << x1 << ", " << y1 << "], [" << x2 << ", " << y2 << "]" << slog::endl;
                cv::Mat mask_mat(H, W, CV_32FC1, mask_arr);

                cv::Rect roi = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), box_width, box_height);
                cv::Mat roi_input_img = output_images[batch](roi);
                const float alpha = 0.7f;

                cv::Mat resized_mask_mat(box_height, box_width, CV_32FC1);
                cv::resize(mask_mat, resized_mask_mat, cv::Size(box_width, box_height));

                cv::Mat uchar_resized_mask(box_height, box_width, CV_8UC3,
                    cv::Scalar(color.blue(), color.green(), color.red()));
                roi_input_img.copyTo(uchar_resized_mask, resized_mask_mat <= MASK_THRESHOLD);

                cv::addWeighted(uchar_resized_mask, alpha, roi_input_img, 1.0 - alpha, 0.0f, roi_input_img);
                cv::rectangle(output_images[batch], roi, cv::Scalar(0, 0, 255), 1);
            }
        }

        metrics.update(startTime);

        for (size_t i = 0; i < output_images.size(); i++) {
            std::string imgName = "out" + std::to_string(i) + ".png";
            cv::imwrite(imgName, output_images[i]);
            slog::info << "Image " << imgName << " created!" << slog::endl;
        }

        slog::info << "Metrics report:" << slog::endl;
        slog::info << "\tLatency: " << std::fixed << std::setprecision(1) << metrics.getTotal().latency << " ms" << slog::endl;
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
