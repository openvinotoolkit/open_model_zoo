// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point for inference engine Mask RCNN demo application
 * @file mask_rcnn_demo/main.cpp
 * @example mask_rcnn_demo/main.cpp
 */
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <map>
#include <algorithm>
#include <string>
#include <vector>
#include <iomanip>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "mask_rcnn_demo.h"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
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

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

int main(int argc, char *argv[]) {
    try {
        std::cout << "InferenceEngine: " << *InferenceEngine::GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imagePaths;
        parseInputFilesArguments(imagePaths);
        if (imagePaths.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // ---------------------Load inference engine------------------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            auto extension_ptr = make_so_pointer<InferenceEngine::IExtension>(FLAGS_l);
            ie.AddExtension(extension_ptr, "CPU");
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "CPU");
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        /** Printing version **/
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        // -----------------------------------------------------------------------------------------------------

        // --------------------Load network (Generated xml/bin files)-------------------------------------------
        slog::info << "Loading network files" << slog::endl;

        /** Read network model **/
        auto network = ie.ReadNetwork(FLAGS_m);

        // add DetectionOutput layer as output so we can get detected boxes and their probabilities
        network.addOutput(FLAGS_detection_output_name.c_str(), 0);
        // -----------------------------------------------------------------------------------------------------

        // -----------------------------Prepare input blobs-----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());

        std::string imageInputName;

        for (const auto & inputInfoItem : inputInfo) {
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // first input contains images
                imageInputName = inputInfoItem.first;
                inputInfoItem.second->setPrecision(Precision::U8);
            } else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // second input contains image info
                inputInfoItem.second->setPrecision(Precision::FP32);
            } else {
                throw std::logic_error("Unsupported input shape with size = " + std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()));
            }
        }

        /** network dimensions for image input **/
        const TensorDesc& inputDesc = inputInfo[imageInputName]->getTensorDesc();
        IE_ASSERT(inputDesc.getDims().size() == 4);
        size_t netBatchSize = getTensorBatch(inputDesc);
        size_t netInputHeight = getTensorHeight(inputDesc);
        size_t netInputWidth = getTensorWidth(inputDesc);

        slog::info << "Network batch size is " << netBatchSize << slog::endl;

        /** Collect images **/
        std::vector<cv::Mat> images;

        if (netBatchSize > imagePaths.size()) {
            slog::warn << "Network batch size is greater than number of images (" << imagePaths.size() <<
                       "), some input files will be duplicated" << slog::endl;
        } else if (netBatchSize < imagePaths.size()) {
            slog::warn << "Network batch size is less than number of images (" << imagePaths.size() <<
                       "), some input files will be ignored" << slog::endl;
        }

        for (size_t i = 0, inputIndex = 0; i < netBatchSize; i++, inputIndex++) {
            if (inputIndex >= imagePaths.size()) {
                inputIndex = 0;
            }
            slog::info << "Prepare image " << imagePaths[inputIndex] << slog::endl;

            cv::Mat image = cv::imread(imagePaths[inputIndex], cv::IMREAD_COLOR);

            if (image.empty()) {
                slog::warn << "Image " + imagePaths[inputIndex] + " cannot be read!" << slog::endl;
                continue;
            }

            images.push_back(image);
        }
        if (images.empty()) throw std::logic_error("Valid input images were not found!");

        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Prepare output blobs------------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
        for (auto & item : outputInfo) {
            item.second->setPrecision(Precision::FP32);
        }

        // -----------------------------------------------------------------------------------------------------

        // -------------------------Load model to the device----------------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        auto executable_network = ie.LoadNetwork(network, FLAGS_d);

        // -------------------------Create Infer Request--------------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        auto infer_request = executable_network.CreateInferRequest();

        // -----------------------------------------------------------------------------------------------------

        // -------------------------------Set input data--------------------------------------------------------
        slog::info << "Setting input data to the blobs" << slog::endl;

        /** Iterate over all the input blobs **/
        for (const auto & inputInfoItem : inputInfo) {
            Blob::Ptr input = infer_request.GetBlob(inputInfoItem.first);

            /** Fill first input tensor with images. First b channel, then g and r channels **/
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {
                /** Iterate over all input images **/
                for (size_t image_id = 0; image_id < images.size(); ++image_id)
                    matU8ToBlob<unsigned char>(images[image_id], input, image_id);
            }

            /** Fill second input tensor with image info **/
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {
                LockedMemory<void> inputMapped = as<MemoryBlob>(input)->wmap();
                auto data = inputMapped.as<float *>();
                data[0] = static_cast<float>(netInputHeight);  // height
                data[1] = static_cast<float>(netInputWidth);  // width
                data[2] = 1;
            }
        }

        // -----------------------------------------------------------------------------------------------------


        // ----------------------------Do inference-------------------------------------------------------------
        slog::info << "Start inference" << slog::endl;
        infer_request.Infer();
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Postprocess output blobs--------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        const auto do_blob = infer_request.GetBlob(FLAGS_detection_output_name.c_str());
        LockedMemory<const void> doBlobMapped = as<MemoryBlob>(do_blob)->rmap();
        const auto do_data  = doBlobMapped.as<float*>();

        const auto masks_blob = infer_request.GetBlob(FLAGS_masks_name.c_str());
        LockedMemory<const void> masksBlobMapped = as<MemoryBlob>(masks_blob)->rmap();
        const auto masks_data = masksBlobMapped.as<float*>();

        const float PROBABILITY_THRESHOLD = 0.2f;
        const float MASK_THRESHOLD = 0.5f;  // threshold used to determine whether mask pixel corresponds to object or to background
        // amount of elements in each detected box description (batch, label, prob, x1, y1, x2, y2)
        IE_ASSERT(do_blob->getTensorDesc().getDims().size() == 2);
        size_t BOX_DESCRIPTION_SIZE = do_blob->getTensorDesc().getDims().back();

        const TensorDesc& masksDesc = masks_blob->getTensorDesc();
        IE_ASSERT(masksDesc.getDims().size() == 4);
        size_t BOXES = getTensorBatch(masksDesc);
        size_t C = getTensorChannels(masksDesc);
        size_t H = getTensorHeight(masksDesc);
        size_t W = getTensorWidth(masksDesc);


        size_t box_stride = W * H * C;

        std::map<size_t, size_t> class_color;

        std::vector<cv::Mat> output_images;
        for (const auto &img : images) {
            output_images.push_back(img.clone());
        }

        /** Iterating over all boxes **/
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
            int box_width = std::min(static_cast<int>(std::max(0.0f, x2 - x1)), images[batch].cols);
            int box_height = std::min(static_cast<int>(std::max(0.0f, y2 - y1)), images[batch].rows);
            auto class_id = static_cast<size_t>(box_info[1] + 1e-6f);
            if (prob > PROBABILITY_THRESHOLD) {
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
        for (size_t i = 0; i < output_images.size(); i++) {
            std::string imgName = "out" + std::to_string(i) + ".png";
            cv::imwrite(imgName, output_images[i]);
            slog::info << "Image " << imgName << " created!" << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl << "This demo is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool from the openVINO toolkit" << slog::endl;
    return 0;
}
