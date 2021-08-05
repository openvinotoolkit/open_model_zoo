// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>

#include "mri_reconstruction_demo.hpp"
#include "npy_reader.hpp"

bool ParseAndCheckCommandLine(int argc, char *argv[]);

static cv::Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob);

struct MRIData {
    cv::Mat data;
    cv::Mat samplingMask;
    cv::Mat reconstructed;
    // Hybrid-CS-Model-MRI/Data/stats_fs_unet_norm_20.npy
    static constexpr double stats[]{2.20295299e-1, 1.11048916e+3};
};

static const std::string kWinName = "MRI reconstruction with OpenVINO";

void callback(int pos, void* userdata);

cv::Mat kspaceToImage(const cv::Mat& kspace);

float psnr(const cv::Mat& data0, const cv::Mat& data1, float maxVal);

int main(int argc, char** argv) {
    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    InferenceEngine::Core ie;

    InferenceEngine::CNNNetwork net = ie.ReadNetwork(FLAGS_m);
    net.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);

    InferenceEngine::ExecutableNetwork execNet = ie.LoadNetwork(net, FLAGS_d);
    InferenceEngine::InferRequest infReq = execNet.CreateInferRequest();

    // Hybrid-CS-Model-MRI/Data/sampling_mask_20perc.npy
    MRIData mri;
    mri.samplingMask = blobFromNPY(FLAGS_p);
    std::cout << "Sampling ratio: " << 1.0 - cv::mean(mri.samplingMask)[0] << std::endl;

    mri.data = blobFromNPY(FLAGS_i);
    CV_Assert(mri.data.depth() == CV_64F);
    const int numSlices = mri.data.size[0];
    const int height = mri.data.size[1];
    const int width = mri.data.size[2];
    mri.data /= sqrt(height * width);

    mri.reconstructed.create({numSlices, height, width}, CV_8U);

    std::cout << "Compute..." << std::endl;

    cv::Mat inputBlob = infEngineBlobToMat(infReq.GetBlob(net.getInputsInfo().begin()->first));
    cv::Mat outputBlob = infEngineBlobToMat(infReq.GetBlob(net.getOutputsInfo().begin()->first));
    outputBlob = outputBlob.reshape(1, height);

    cv::TickMeter tm;
    tm.start();
    for (int i = 0; i < numSlices; ++i) {
        // Prepare input
        cv::Mat kspace = cv::Mat(height, width, CV_64FC2, mri.data.ptr<double>(i)).clone();

        kspace.setTo(0, mri.samplingMask);
        kspace = (kspace - cv::Scalar(mri.stats[0], mri.stats[0])) / cv::Scalar(mri.stats[1], mri.stats[1]);
        kspace.reshape(1, 1).convertTo(inputBlob.reshape(1, 1), CV_32F);

        // Forward pass
        infReq.Infer();

        // Save prediction
        cv::Mat slice(height, width, CV_8UC1, mri.reconstructed.ptr<uint8_t>(i));
        cv::normalize(outputBlob, slice, 255, 0, cv::NORM_MINMAX, CV_8U);
    }
    tm.stop();
    std::cout << cv::format("Elapsed time: %.1f seconds", tm.getTimeSec()) << std::endl;

    // Visualization loop.
    int sliceId = numSlices / 2;
    cv::namedWindow(kWinName, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Slice", kWinName, nullptr, numSlices - 1, callback, &mri);
    callback(sliceId, &mri);  // Trigger initial visualization
    cv::waitKey();
    return 0;
}

cv::Mat infEngineBlobToMat(const InferenceEngine::Blob::Ptr& blob) {
    // NOTE: Inference Engine sizes are reversed.
    std::vector<size_t> dims = blob->getTensorDesc().getDims();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob->getTensorDesc().getPrecision();
    CV_Assert(precision == InferenceEngine::Precision::FP32);
    return cv::Mat(size, CV_32F, (void*)blob->buffer());
}

void callback(int sliceId, void* userdata) {
    MRIData* mri = reinterpret_cast<MRIData*>(userdata);
    const int height = mri->data.size[1];
    const int width = mri->data.size[2];

    cv::Mat kspace = cv::Mat(height, width, CV_64FC2, mri->data.ptr<double>(sliceId)).clone();
    cv::Mat img = kspaceToImage(kspace);

    kspace.setTo(0, mri->samplingMask);
    cv::Mat masked = kspaceToImage(kspace);

    cv::Mat rec(height, width, CV_8U, mri->reconstructed.ptr<uint8_t>(sliceId));

    static const int kBorderSize = 20;
    cv::Mat render;
    cv::hconcat(std::vector<cv::Mat>({img, masked, rec}), render);
    cv::copyMakeBorder(render, render, kBorderSize, 0, 0, 0, cv::BORDER_CONSTANT, 255);
    cv::putText(render, "Original", cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, 0);
    cv::putText(render, cv::format("Sampled (PSNR %.1f)", cv::PSNR(img, masked, 255)),
                cv::Point(width, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, 0);
    cv::putText(render, cv::format("Reconstructed (PSNR %.1f)", cv::PSNR(img, rec, 255)),
                cv::Point(width*2, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, 0);

    cv::imshow(kWinName, render);
    cv::waitKey(1);
}

cv::Mat kspaceToImage(const cv::Mat& kspace) {
    CV_CheckEQ(kspace.dims, 2, "");
    CV_CheckEQ(kspace.channels(), 2, "");

    cv::Mat img;

    cv::idft(kspace, img, cv::DFT_SCALE);

    std::vector<cv::Mat> components;
    cv::split(img, components);
    cv::magnitude(components[0], components[1], img);
    cv::normalize(img, img, 255, 0, cv::NORM_MINMAX, CV_8U);
    return img;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_p.empty()) {
        throw std::logic_error("Parameter -p is not set");
    }

    return true;
}
