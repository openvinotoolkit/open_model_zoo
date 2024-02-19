// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point for Mask RCNN demo application
 * @file mask_rcnn_demo/main.cpp
 * @example mask_rcnn_demo/main.cpp
 */
#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"

#include "gflags/gflags.h"
#include "utils/common.hpp"
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

void validateInputsAndOutputs(ov::OutputVector& inputs, ov::OutputVector& outputs) {
    if (inputs.size() != 2 || outputs.size() != 2)
        throw std::logic_error("Expected model with 2 inputs and 2 outputs");

    std::string input_error = "Unexpected input dimensions: ";
    std::string output_error = "Unexpected output dimensions: ";

    auto check_dims = [&](ov::OutputVector& nodes, std::string& error_msg) {
        for (ov::Output<ov::Node> node : nodes) {
            size_t dims = node.get_shape().size();
            if (dims != 2 && dims != 4)
                throw std::logic_error(error_msg + std::to_string(dims));
        }
    };

    check_dims(inputs, input_error);
    check_dims(outputs, output_error);
}

int main(int argc, char* argv[]) {
    std::set_terminate(catcher);

    PerformanceMetrics metrics;

    // Parsing and validation of input args
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    // This vector stores paths to the processed images
    std::vector<std::string> imagePaths;
    parseInputFilesArguments(imagePaths);
    if (imagePaths.empty())
        throw std::runtime_error("No suitable images were found");

    // Load OpenVINO runtime
    slog::info << ov::get_openvino_version() << slog::endl;
    ov::Core core;

    // Load model (Generated xml/bin files)

    // Read model
    slog::info << "Reading model: " << FLAGS_m << slog::endl;
    std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
    logBasicModelInfo(model);

    // Taking information about model inputs
    ov::OutputVector inputs = model->inputs();
    ov::OutputVector outputs = model->outputs();

    // Validate inputs/outputs
    validateInputsAndOutputs(inputs, outputs);

    size_t modelBatchSize = 0;

    std::string image_tensor_name;
    size_t image_tensor_width = 0;
    size_t image_tensor_height = 0;
    ov::Layout image_tensor_layout;
    ov::Shape image_tensor_shape;
    ov::element::Type image_tensor_type;

    std::string info_tensor_name;
    ov::Layout info_tensor_layout = {"?C"};
    ov::Shape info_tensor_shape;

    for (ov::Output<ov::Node> input : inputs) {
        ov::Shape shape = input.get_shape();
        size_t dims = shape.size();

        if (dims == 2) {
            info_tensor_name = input.get_any_name();
        }
        if (dims == 4) {
            image_tensor_name = input.get_any_name();
            image_tensor_type = input.get_element_type();
            image_tensor_shape = shape;
            image_tensor_layout = ov::layout::get_layout(input);
            if (image_tensor_layout.empty() && image_tensor_type == ov::element::f32)
                image_tensor_layout = {"NCHW"};
            if (image_tensor_layout.empty() && image_tensor_type == ov::element::u8)
                image_tensor_layout = {"NHWC"};
            image_tensor_height = image_tensor_shape[ov::layout::height_idx(image_tensor_layout)];
            image_tensor_width = image_tensor_shape[ov::layout::width_idx(image_tensor_layout)];
            modelBatchSize = image_tensor_shape[ov::layout::batch_idx(image_tensor_layout)];
        }
    }

    ov::Layout desired_tensor_layout = {"NHWC"};

    std::string boxes_tensor_name;
    std::string masks_tensor_name;
    ov::Layout masks_tensor_layout = {"NCHW"}; // expected by mask processing logic

    for (ov::Output<ov::Node> output : outputs) {
        ov::Shape shape = output.get_shape();
        size_t dims = shape.size();
        if (dims == 2) {
            boxes_tensor_name = output.get_any_name();
        }
        if (dims == 4) {
            masks_tensor_name = output.get_any_name();
        }
    }

    // Collect images
    std::vector<cv::Mat> images;

    if (modelBatchSize > imagePaths.size()) {
        slog::warn << "Model batch size is greater than number of images (" << imagePaths.size() <<
            "), some input files will be duplicated" << slog::endl;
    }
    else if (modelBatchSize < imagePaths.size()) {
        modelBatchSize = imagePaths.size();
        slog::warn << "Model batch size is less than number of images (" << imagePaths.size() <<
            "), model will be reshaped" << slog::endl;
    }

    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input(image_tensor_name)
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout(desired_tensor_layout);
    ppp.input(info_tensor_name)
        .tensor()
        .set_layout(info_tensor_layout);

    ppp.input(image_tensor_name).model()
        .set_layout(image_tensor_layout);

    model = ppp.build();
    slog::info << "Preprocessor configuration: " << slog::endl;
    slog::info << ppp << slog::endl;

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

    // Set image input data
    for (size_t idx = 0; idx < inputs.size(); idx++) {
        ov::Tensor tensor = infer_request.get_input_tensor(idx);
        ov::Shape shape = tensor.get_shape();

        if (shape.size() == 4) {
            for (size_t batchId = 0; batchId < modelBatchSize; ++batchId) {
                cv::Size size = {
                    int(image_tensor_width),
                    int(image_tensor_height)
                };
                unsigned char* data = tensor.data<unsigned char>() + batchId * image_tensor_width * image_tensor_height * 3;
                cv::Mat image_resized(size, CV_8UC3, data);
                cv::resize(images[batchId], image_resized, size);
            }
        }

        if (shape.size() == 2) {
            float* data = tensor.data<float>();
            data[0] = static_cast<float>(image_tensor_height); // height
            data[1] = static_cast<float>(image_tensor_width); // width
            data[2] = 1;
        }
    }

    slog::info << "Start inference..." << slog::endl;
    // Do inference
    infer_request.infer();

    // Postprocess output data
    float* boxes_data = nullptr;
    float* masks_data = nullptr;

    size_t BOX_DESCRIPTION_SIZE = 0;

    size_t BOXES = 0;
    size_t C = 0;
    size_t H = 0;
    size_t W = 0;

    slog::info << "Processing results..." << slog::endl;
    for (size_t idx = 0; idx < outputs.size(); idx++) {
        ov::Tensor tensor = infer_request.get_output_tensor(idx);
        ov::Shape shape = tensor.get_shape();
        size_t dims = shape.size();
        if (dims == 2) {
            boxes_data = tensor.data<float>();
            // amount of elements in each detected box description (batch, label, prob, x1, y1, x2, y2)
            BOX_DESCRIPTION_SIZE = shape[1];
        }
        if (dims == 4) {
            masks_data = tensor.data<float>();
            BOXES = shape[ov::layout::batch_idx(masks_tensor_layout)];
            C = shape[ov::layout::channels_idx(masks_tensor_layout)];
            H = shape[ov::layout::height_idx(masks_tensor_layout)];
            W = shape[ov::layout::width_idx(masks_tensor_layout)];
        }
    }

    const float PROBABILITY_THRESHOLD = 0.2f;
    // threshold used to determine whether mask pixel corresponds to object or to background
    const float MASK_THRESHOLD = 0.5f;

    size_t box_stride = W * H * C;

    std::map<size_t, size_t> class_color;

    std::vector<cv::Mat> output_images;
    for (const auto& img : images) {
        output_images.push_back(img.clone());
    }

    // Iterating over all boxes
    for (size_t box = 0; box < BOXES; ++box) {
        float* box_info = boxes_data + box * BOX_DESCRIPTION_SIZE;
        auto batch = static_cast<int>(box_info[0]);

        if (batch < 0)
            break;
        if (batch >= static_cast<int>(modelBatchSize))
            throw std::logic_error("Invalid batch ID within detection output box");

        float prob = box_info[2];

        float x1 = clamp(static_cast<float>(images[batch].cols), .0f, box_info[3] * images[batch].cols);
        float y1 = clamp(static_cast<float>(images[batch].rows), .0f, box_info[4] * images[batch].rows);
        float x2 = clamp(static_cast<float>(images[batch].cols), .0f, box_info[5] * images[batch].cols);
        float y2 = clamp(static_cast<float>(images[batch].rows), .0f, box_info[6] * images[batch].rows);

        int box_width = static_cast<int>(x2 - x1);
        int box_height = static_cast<int>(y2 - y1);

        size_t class_id = static_cast<size_t>(box_info[1] + 1e-6);

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
        if(!cv::imwrite(imgName, output_images[i]))
            throw std::runtime_error("Can't write image to file: " + imgName);
        slog::info << "Image " << imgName << " created!" << slog::endl;
    }

    slog::info << "Metrics report:" << slog::endl;
    slog::info << "\tLatency: " << std::fixed << std::setprecision(1) << metrics.getTotal().latency << " ms" << slog::endl;

    return 0;
}
