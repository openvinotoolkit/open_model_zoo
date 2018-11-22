/*
// Copyright (c) 2018 Intel Corporation
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

/**
 * @brief The entry point for inference engine Mask RCNN demo application
 * @file mask_rcnn_demo/main.cpp
 * @example mask_rcnn_demo/main.cpp
 */
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <memory>
#include <map>
#include <algorithm>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <time.h>
#include <chrono>
#include <limits>
#include <iomanip>

#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "mask_rcnn_demo.h"
#include <opencv2/opencv.hpp>

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_ni < 1) {
        throw std::logic_error("Parameter -ni should be greater than 0 (default: 1)");
    }

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
        std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        if (images.empty()) throw std::logic_error("No suitable images were found");
        if (images.size() > 1) throw std::logic_error("The topology support only one input image");
        // -----------------------------------------------------------------------------------------------------

        // ---------------------Load plugin for inference engine------------------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp, "../../../lib/intel64" , "" }).getPluginByDevice(FLAGS_d);

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            auto extension_ptr = make_so_pointer<InferenceEngine::IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        /** Setting plugin parameter for per layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        /** Printing plugin version **/
        printPluginVersion(plugin, std::cout);

        // -----------------------------------------------------------------------------------------------------

        // --------------------Load network (Generated xml/bin files)-------------------------------------------
        slog::info << "Loading network files" << slog::endl;

        InferenceEngine::CNNNetReader networkReader;
        /** Read network model **/
        networkReader.ReadNetwork(FLAGS_m);

        /** Extract model name and load weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        networkReader.ReadWeights(binFileName);
        auto network = networkReader.getNetwork();

        // add DetectionOutput layer as output so we can get detected boxes and their probabilities
        network.addOutput(FLAGS_detection_output_name.c_str(), 0);
        // -----------------------------------------------------------------------------------------------------

        // -----------------------------Prepare input blobs-----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());
        /** Stores all input blobs data **/
        InferenceEngine::BlobMap inputBlobs;

        if (inputInfo.size() != 1) throw std::logic_error("Demo supports topologies only with 1 input");
        auto inputInfoItem = *inputInfo.begin();

        /** Collect images data ptrs **/
        std::vector<std::shared_ptr<unsigned char>> imagesData;
        std::vector<cv::Mat> images_cv;
        for (auto & i : images) {
            images_cv.push_back(cv::imread(i, cv::IMREAD_COLOR));
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Getting image data **/
            std::shared_ptr<unsigned char> data(reader->getData(inputInfoItem.second->getDims()[0], inputInfoItem.second->getDims()[1]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        /** Setting batch size to 1 since it is the only supported batch **/
        network.setBatchSize(1);
        slog::info << "Batch size is " << std::to_string(networkReader.getNetwork().getBatchSize()) << slog::endl;

        inputInfoItem.second->setPrecision(Precision::U8);

        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Prepare output blobs------------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());
        for (auto & item : outputInfo) {
            item.second->setPrecision(Precision::FP32);
        }
        // -----------------------------------------------------------------------------------------------------

        // -------------------------Load model to the plugin-------------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;

        auto executable_network = plugin.LoadNetwork(network, {});
        auto infer_request = executable_network.CreateInferRequest();

        // -----------------------------------------------------------------------------------------------------

        // -------------------------------Set input data----------------------------------------------------
        /** Iterate over all the input blobs **/
        /** Iterating over all input blobs **/
        for (const auto & item : inputInfo) {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);

            /** Fill input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = input->dims()[2];
            size_t image_size = input->dims()[1] * input->dims()[0];

            auto data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------


        // ----------------------------Do inference-------------------------------------------------------------
        slog::info << "Start inference (" << FLAGS_ni << " iterations)" << slog::endl;

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

        double total = 0.0;
        /** Start inference & calc performance **/
        for (int iter = 0; iter < FLAGS_ni; ++iter) {
            auto t0 = Time::now();
            infer_request.Infer();
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
        }

        /** Show performance results **/
        std::cout << std::endl << "Average running time of one iteration: " << total / static_cast<double>(FLAGS_ni) << " ms" << std::endl << std::endl;

        if (FLAGS_pc) {
            printPerformanceCounts(infer_request, std::cout);
        }
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Postprocess output blobs--------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        const auto do_blob = infer_request.GetBlob(FLAGS_detection_output_name.c_str());
        const auto do_data = do_blob->buffer().as<float*>();

        const auto masks_blob = infer_request.GetBlob(FLAGS_masks_name.c_str());
        const auto masks_data = masks_blob->buffer().as<float*>();

        const float PROBABILITY_THRESHOLD = 0.2;
        const float MASK_THRESHOLD = 0.5;  // threshold used to determine whever mask pixel corresponds to object or to background
        size_t BOX_DESCRIPTION_SIZE = do_blob->dims().at(0);  // amount of elements in each detected box description (batch, label, prob, x1, y1, x2, y2)
        size_t BOXES = masks_blob->dims().at(3);
        size_t C = masks_blob->dims().at(2);
        size_t H = masks_blob->dims().at(1);
        size_t W = masks_blob->dims().at(0);

        size_t box_stride = W * H * C;

        // some colours
        std::vector<std::vector<short>> colors = {
            {128, 64,  128},
            {232, 35,  244},
            {70,  70,  70},
            {156, 102, 102},
            {153, 153, 190},
            {153, 153, 153},
            {30,  170, 250},
            {0,   220, 220},
            {35,  142, 107},
            {152, 251, 152},
            {180, 130, 70},
            {60,  20,  220},
            {0,   0,   255},
            {142, 0,   0},
            {70,  0,   0},
            {100, 60,  0},
            {90,  0,   0},
            {230, 0,   0},
            {32,  11,  119},
            {0,   74,  111},
            {81,  0,   81}
        };
        std::map<int, int> class_color;

        cv::Mat output_image = images_cv[0].clone();
        /** Iterating over all boxes **/
        for (size_t box = 0; box < BOXES; ++box) {
            float* box_info = do_data + box * BOX_DESCRIPTION_SIZE;
            float batch = box_info[0];
            if (batch < 0)
                break;
            float prob = box_info[2];
            float x1 = std::min(std::max(0.0f, box_info[3] * images_cv[0].size().width), static_cast<float>(images_cv[0].size().width));
            float y1 = std::min(std::max(0.0f, box_info[4] * images_cv[0].size().height), static_cast<float>(images_cv[0].size().height));
            float x2 = std::min(std::max(0.0f, box_info[5] * images_cv[0].size().width), static_cast<float>(images_cv[0].size().width));
            float y2 = std::min(std::max(0.0f, box_info[6] * images_cv[0].size().height), static_cast<float>(images_cv[0].size().height));
            int box_width = std::min(static_cast<int>(std::max(0.0f, x2 - x1)), images_cv[0].size().width);
            int box_height = std::min(static_cast<int>(std::max(0.0f, y2 - y1)), images_cv[0].size().height);
            int class_id = static_cast<int>(box_info[1] + 1e-6f);
            if (prob > PROBABILITY_THRESHOLD) {
                if (class_color.find(class_id) == class_color.end())
                    class_color[class_id] = class_color.size();
                auto& color = colors[class_color[class_id]];
                float* mask_arr = masks_data + box_stride * box + H * W * (class_id - 1);
                slog::info << "Detected class " << class_id << " with probability " << prob << ": [" << x1
                           << ", " << y1 << "], [" << x2 << ", " << y2 << "]" << slog::endl;
                cv::Mat mask_mat(H, W, CV_32FC1, mask_arr);

                cv::Rect roi = cv::Rect(static_cast<int>(x1), static_cast<int>(y1), box_width, box_height);
                cv::Mat roi_input_img = output_image(roi);
                const float alpha = 0.7f;

                cv::Mat resized_mask_mat(box_height, box_width, CV_32FC1);
                cv::resize(mask_mat, resized_mask_mat, cv::Size(box_width, box_height));

                cv::Mat uchar_resized_mask(box_height, box_width, images_cv[0].type());

                for (int h = 0; h < resized_mask_mat.size().height; ++h)
                    for (int w = 0; w < resized_mask_mat.size().width; ++w)
                        for (int ch = 0; ch < uchar_resized_mask.channels(); ++ch)
                            uchar_resized_mask.at<cv::Vec3b>(h, w)[ch] = resized_mask_mat.at<float>(h, w) > MASK_THRESHOLD ?
                                                                            255 * color[ch]: roi_input_img.at<cv::Vec3b>(h, w)[ch];

                cv::addWeighted(uchar_resized_mask, alpha, roi_input_img, 1.0f - alpha, 0.0f, roi_input_img);
                cv::rectangle(output_image, roi, cv::Scalar(0, 0, 1), 1);
            }
        }
        cv::imwrite("out.png", output_image);
        slog::info << "Image out.png created!" << slog::endl;
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
    return 0;
}
