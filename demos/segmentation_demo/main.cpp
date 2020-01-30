// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <opencv2/videoio.hpp>

#include <inference_engine.hpp>

#include <monitors/presenter.h>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "segmentation_demo.h"

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

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

int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        InferenceEngine::Core ie;

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            InferenceEngine::IExtensionPtr extension_ptr =
                InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(FLAGS_l);
            ie.AddExtension(extension_ptr, "CPU");
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        slog::info << "Device info" << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        InferenceEngine::CNNNetwork network = ie.ReadNetwork(FLAGS_m);

        InferenceEngine::ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
        if (inputShapes.size() != 1)
            throw std::runtime_error("Demo supports topologies only with 1 input");
        const std::string& inName = inputShapes.begin()->first;
        InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
        if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
            throw std::runtime_error("3 channel 4 dimentional model's input is expected");
        inSizeVector[0] = 1;  // set batch size to 1
        network.reshape(inputShapes);

        InferenceEngine::InputInfo& inputInfo = *network.getInputsInfo().begin()->second;
        inputInfo.getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        inputInfo.setLayout(InferenceEngine::Layout::NHWC);
        inputInfo.setPrecision(InferenceEngine::Precision::U8);

        const InferenceEngine::OutputsDataMap& outputsDataMap = network.getOutputsInfo();
        if (outputsDataMap.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");
        const std::string& outName = outputsDataMap.begin()->first;
        InferenceEngine::Data& data = *outputsDataMap.begin()->second;
        // alternative to resetting the output precision is to have a switch
        // statement in postprocessing handling other precision types
        data.setPrecision(InferenceEngine::Precision::FP32);
        const InferenceEngine::SizeVector& outSizeVector = data.getTensorDesc().getDims();
        int outChannels, outHeight, outWidth;
        switch(outSizeVector.size()) {
            case 3:
                outChannels = 0;
                outHeight = outSizeVector[1];
                outWidth = outSizeVector[2];
                break;
            case 4:
                outChannels = outSizeVector[1];
                outHeight = outSizeVector[2];
                outWidth = outSizeVector[3];
                break;
            default:
                throw std::runtime_error("Unexpected output blob shape. Only 4D and 3D output blobs are"
                    "supported.");
        }

        InferenceEngine::ExecutableNetwork executableNetwork = ie.LoadNetwork(network, FLAGS_d);
        InferenceEngine::InferRequest inferRequest = executableNetwork.CreateInferRequest();

        cv::VideoCapture cap;
        try {
            int index = std::stoi(FLAGS_i);
            if (!cap.open(index))
                throw std::runtime_error("Can't open camera " + std::to_string(index));
        } catch (const std::invalid_argument&) {
            if (!cap.open(FLAGS_i))
                throw std::runtime_error("Can't open input " + FLAGS_i);
        } catch (const std::out_of_range&) {
            if (!cap.open(FLAGS_i))
                throw std::runtime_error("Can't open input " + FLAGS_i);
        }

        float blending = 0.3f;
        constexpr char WIN_NAME[] = "segmentation";
        if (!FLAGS_no_show) {
            cv::namedWindow(WIN_NAME);
            int initValue = static_cast<int>(blending * 100);
            cv::createTrackbar("blending", WIN_NAME, &initValue, 100,
                [](int position, void* blendingPtr){*static_cast<float*>(blendingPtr) = position * 0.01f;},
                &blending);
        }

        cv::Mat inImg, resImg, maskImg(outHeight, outWidth, CV_8UC3);
        std::vector<cv::Vec3b> colors(arraySize(CITYSCAPES_COLORS));
        for (std::size_t i = 0; i < colors.size(); ++i)
            colors[i] = {CITYSCAPES_COLORS[i].blue(), CITYSCAPES_COLORS[i].green(), CITYSCAPES_COLORS[i].red()};
        std::mt19937 rng;
        std::uniform_int_distribution<uchar> distr(0, 255);
        int delay = 1;
        cv::Size graphSize{static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60};
        Presenter presenter(FLAGS_u, 10, graphSize);

        while (cap.read(inImg)) {
            if (CV_8UC3 != inImg.type())
                throw std::runtime_error("BGR (or RGB) image expected to come from input");
            inferRequest.SetBlob(inName, wrapMat2Blob(inImg));
            inferRequest.Infer();

            const float * const predictions = inferRequest.GetBlob(outName)->cbuffer().as<float*>();
            for (int rowId = 0; rowId < outHeight; ++rowId) {
                for (int colId = 0; colId < outWidth; ++colId) {
                    std::size_t classId = 0;
                    if (outChannels == 0) {  // assume the output is already ArgMax'ed
                        classId = predictions[rowId * outWidth + colId];
                    } else {
                        float maxProb = -1.0f;
                        for (int chId = 0; chId < outChannels; ++chId) {
                            float prob = predictions[chId * outHeight * outWidth + rowId * outWidth + colId];
                            if (prob > maxProb) {
                                classId = chId;
                                maxProb = prob;
                            }
                        }
                    }
                    while (classId >= colors.size()) {
                        cv::Vec3b color(distr(rng), distr(rng), distr(rng));
                        colors.push_back(color);
                    }
                    maskImg.at<cv::Vec3b>(rowId, colId) = colors[classId];
                }
            }
            cv::resize(maskImg, resImg, inImg.size());
            resImg = inImg * blending + resImg * (1 - blending);
            presenter.drawGraphs(resImg);
            if (!FLAGS_no_show) {
                cv::imshow(WIN_NAME, resImg);
                int key = cv::waitKey(delay);
                switch(key) {
                    case 'q':
                    case 'Q':
                    case 27: // Esc
                        delay = -1;
                        break;
                    case 'p':
                    case 'P':
                    case 32: // Space
                        delay = !delay;
                        break;
                    default:
                        presenter.handleKey(key);
                }
            }
            if (delay == -1)
                break;
        }
        std::cout << presenter.reportMeans() << '\n';
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
