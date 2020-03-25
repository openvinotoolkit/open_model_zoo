// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
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

using namespace InferenceEngine;
typedef std::chrono::duration<double, std::chrono::milliseconds::period> Ms;

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
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        Core ie;

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            ie.AddExtension(extension_ptr, "CPU");
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        slog::info << "Device info" << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        CNNNetwork network = ie.ReadNetwork(FLAGS_m);

        ICNNNetwork::InputShapes inputShapes = network.getInputShapes();
        if (inputShapes.size() != 1)
            throw std::runtime_error("Demo supports topologies only with 1 input");
        const std::string& inName = inputShapes.begin()->first;
        SizeVector& inSizeVector = inputShapes.begin()->second;
        if (inSizeVector.size() != 4 || inSizeVector[1] != 3)
            throw std::runtime_error("3-channel 4-dimensional model's input is expected");
        inSizeVector[0] = 1;  // set batch size to 1
        network.reshape(inputShapes);

        InputInfo& inputInfo = *network.getInputsInfo().begin()->second;
        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        inputInfo.setLayout(Layout::NHWC);
        inputInfo.setPrecision(Precision::U8);

        const OutputsDataMap& outputsDataMap = network.getOutputsInfo();
        if (outputsDataMap.size() != 1) throw std::runtime_error("Demo supports topologies only with 1 output");
        const std::string& outName = outputsDataMap.begin()->first;
        Data& data = *outputsDataMap.begin()->second;
        // if the model performs ArgMax, its output type can be I32 but for models that return heatmaps for each
        // class the output is usually FP32. Reset the precision to avoid handling different types with switch in
        // postprocessing
        data.setPrecision(Precision::FP32);
        const SizeVector& outSizeVector = data.getTensorDesc().getDims();
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

        ExecutableNetwork executableNetwork = ie.LoadNetwork(network, FLAGS_d);
        InferRequest inferRequest = executableNetwork.CreateInferRequest();

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
        std::uniform_int_distribution<int> distr(0, 255);
        int delay = FLAGS_delay;
        cv::Size graphSize{static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH) / 4), 60};
        Presenter presenter(FLAGS_u, 10, graphSize);

        std::chrono::steady_clock::duration latencySum{0};
        unsigned latencySamplesNum = 0;
        std::ostringstream latencyStream;

        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        while (cap.read(inImg) && delay >= 0) {
            if (CV_8UC3 != inImg.type())
                throw std::runtime_error("BGR (or RGB) image expected to come from input");
            inferRequest.SetBlob(inName, wrapMat2Blob(inImg));
            inferRequest.Infer();

            const float * const predictions = inferRequest.GetBlob(outName)->cbuffer().as<float*>();
            for (int rowId = 0; rowId < outHeight; ++rowId) {
                for (int colId = 0; colId < outWidth; ++colId) {
                    std::size_t classId = 0;
                    if (outChannels == 0) {  // assume the output is already ArgMax'ed
                        classId = static_cast<std::size_t>(predictions[rowId * outWidth + colId]);
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

            latencySum += std::chrono::steady_clock::now() - t0;
            ++latencySamplesNum;
            latencyStream.str("");
            latencyStream << std::fixed << std::setprecision(1)
                << (std::chrono::duration_cast<Ms>(latencySum) / latencySamplesNum).count() << " ms";
            constexpr int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
            constexpr double FONT_SCALE = 2;
            constexpr int THICKNESS = 2;
            int baseLine;
            cv::getTextSize(latencyStream.str(), FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
            cv::putText(resImg, latencyStream.str(), cv::Size{0, resImg.rows - baseLine}, FONT_FACE, FONT_SCALE,
                cv::Scalar{255, 0, 0}, THICKNESS);

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
                        delay = !delay * (FLAGS_delay + !FLAGS_delay);
                        break;
                    default:
                        presenter.handleKey(key);
                }
            }
            t0 = std::chrono::steady_clock::now();
        }
        std::cout << "Mean pipeline latency: " << latencyStream.str() << '\n';
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
