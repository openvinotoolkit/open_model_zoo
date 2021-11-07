// Copyright (C) 2018-2019 Intel Corporation
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

#include <gflags/gflags.h>
#include <utils/args_helper.hpp>
#include <utils/ocv_common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "mask_rcnn_demo.h"

using namespace ov::preprocess;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
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

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // This vector stores paths to the processed images
        std::vector<std::string> imagePaths;
        parseInputFilesArguments(imagePaths);
        if (imagePaths.empty())
            throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // ---------------------Load inference engine------------------------------------------------
        slog::info << ov::get_openvino_version() << slog::endl;
        ov::runtime::Core core;

        // -----------------------------------------------------------------------------------------------------

        // --------------------Load network (Generated xml/bin files)-------------------------------------------

        // Read network model
        std::shared_ptr<ov::Function> network = core.read_model(FLAGS_m);
        slog::info << "model file: " << FLAGS_m << slog::endl;
        slog::info << "model name: " << network->get_friendly_name() << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // -----------------------------Prepare input blobs-----------------------------------------------------

        // Taking information about all topology inputs
        ov::OutputVector inputs = network->inputs();
        ov::OutputVector outputs = network->outputs();

        if(inputs.size() != 2 || outputs.size() != 2)
            throw std::logic_error("Expected network with 2 inputs and 2 outputs");

        for (const ov::Output<ov::Node> input : inputs)
        {
            slog::info << slog::endl;
            const std::string name = input.get_any_name();
            const ov::element::Type type = input.get_element_type();
            const ov::Shape shape = input.get_shape();

            slog::info << "input name: " << name << slog::endl;
            slog::info << "input type: " << type << slog::endl;
            slog::info << "input shape: " << shape << slog::endl;
        }

        for (const ov::Output<ov::Node> output : outputs)
        {
            slog::info << slog::endl;
            const std::string name = output.get_any_name();
            const ov::element::Type type = output.get_element_type();
            const ov::Shape shape = output.get_shape();

            slog::info << "output name: " << name << slog::endl;
            slog::info << "output type: " << type << slog::endl;
            slog::info << "output shape: " << shape << slog::endl;
        }

        size_t netBatchSize = 0;
        size_t netInputHeight = 0;
        size_t netInputWidth = 0;

        const ov::Layout layout_nchw{ "NCHW" };

        // network dimensions for image input
        auto it = std::find_if(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& obj) {return obj.get_shape().size() == 4;});
        if (it != inputs.end()) {
            netBatchSize = it->get_shape()[ov::layout::batch_idx(layout_nchw)];
            netInputHeight = it->get_shape()[ov::layout::height_idx(layout_nchw)];
            netInputWidth = it->get_shape()[ov::layout::width_idx(layout_nchw)];
        }

        // Collect images
        std::vector<cv::Mat> images;

        if (netBatchSize > imagePaths.size()) {
            slog::warn << "Network batch size is greater than number of images (" << imagePaths.size() <<
                       "), some input files will be duplicated" << slog::endl;
        } else if (netBatchSize < imagePaths.size()) {
            slog::warn << "Network batch size is less than number of images (" << imagePaths.size() <<
                       "), some input files will be ignored" << slog::endl;
        }

        auto startTime = std::chrono::steady_clock::now();
        for (size_t i = 0, inputIndex = 0; i < netBatchSize; i++, inputIndex++) {
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
        // -----------------------------------------------------------------------------------------------------

        // -----------------------------------------------------------------------------------------------------
/* TODO: not clear yet how to use PrePostProcessor for models with more than one input or output
        network = PrePostProcessor().
            input(InputInfo().
                tensor(InputTensorInfo().
                    set_element_type(ov::element::f32).
//                                  set_spatial_static_shape(
//                                      tensor_shape[ov::layout::height_idx(tensor_layout)],
//                                      tensor_shape[ov::layout::width_idx(tensor_layout)]).
                    set_layout({"NCHW"})).
                preprocess(PreProcessSteps().
                    resize(ResizeAlgorithm::RESIZE_LINEAR)).
                network(InputNetworkInfo().
                    set_layout("NCHW"))).
            output(OutputInfo().
                tensor(OutputTensorInfo().
                    set_element_type(ov::element::f32))).
            build(network);
*/

        // -------------------------Load model to the device----------------------------------------------------
        ov::runtime::ExecutableNetwork executableNetwork = core.compile_model(network, FLAGS_d);
        logExecNetworkInfo(executableNetwork, FLAGS_m, FLAGS_d);
        slog::info << "\tBatch size is set to " << netBatchSize << slog::endl;

        // -------------------------Create Infer Request--------------------------------------------------------
        ov::runtime::InferRequest infer_request = executableNetwork.create_infer_request();
        // -----------------------------------------------------------------------------------------------------

        // -------------------------------Set input data--------------------------------------------------------
        // Iterate over all the input blobs
        for (size_t idx = 0; idx < inputs.size(); idx++) {
            ov::runtime::Tensor tensor = infer_request.get_input_tensor(idx);
            ov::Shape shape = tensor.get_shape();

            if (shape.size() == 4) {
                for (size_t image_id = 0; image_id < images.size(); ++image_id)
                    matToTensor(images[image_id], tensor, image_id);
            }

            if (shape.size() == 2) {
                float* data = tensor.data<float>();
                data[0] = static_cast<float>(netInputHeight);  // height
                data[1] = static_cast<float>(netInputWidth);  // width
                data[2] = 1;
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // ----------------------------Do inference-------------------------------------------------------------
        infer_request.infer();
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Postprocess output blobs--------------------------------------------------
        float* do_data = nullptr;
        float* masks_data = nullptr;

        size_t BOX_DESCRIPTION_SIZE = 0;

        size_t BOXES = 0;
        size_t C = 0;
        size_t H = 0;
        size_t W = 0;

        for (size_t idx = 0; idx < outputs.size(); idx++) {
            ov::runtime::Tensor tensor = infer_request.get_output_tensor(idx);
            ov::Shape shape = tensor.get_shape();
            size_t dims = shape.size();
            if (dims == 2) {
                do_data = tensor.data<float>();
                // amount of elements in each detected box description (batch, label, prob, x1, y1, x2, y2)
                BOX_DESCRIPTION_SIZE = shape[1];
            }
            if (dims == 4) {
                masks_data = tensor.data<float>();
                BOXES = shape[ov::layout::batch_idx(layout_nchw)];
                C = shape[ov::layout::channels_idx(layout_nchw)];
                H = shape[ov::layout::height_idx(layout_nchw)];
                W = shape[ov::layout::width_idx(layout_nchw)];
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
            if (batch >= static_cast<int>(netBatchSize))
                throw std::logic_error("Invalid batch ID within detection output box");

            float prob = box_info[2];

            float x1 = std::min(std::max(0.0f, box_info[3] * images[batch].cols), static_cast<float>(images[batch].cols));
            float y1 = std::min(std::max(0.0f, box_info[4] * images[batch].rows), static_cast<float>(images[batch].rows));
            float x2 = std::min(std::max(0.0f, box_info[5] * images[batch].cols), static_cast<float>(images[batch].cols));
            float y2 = std::min(std::max(0.0f, box_info[6] * images[batch].rows), static_cast<float>(images[batch].rows));

            int box_width = static_cast<int>(x2 - x1);
            int box_height = static_cast<int>(y2 - y1);

            auto class_id = static_cast<size_t>(box_info[1] + 1e-6f);

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

                cv::addWeighted(uchar_resized_mask, alpha, roi_input_img, 1.0f - alpha, 0.0f, roi_input_img);
                cv::rectangle(output_images[batch], roi, cv::Scalar(0, 0, 1), 1);
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
