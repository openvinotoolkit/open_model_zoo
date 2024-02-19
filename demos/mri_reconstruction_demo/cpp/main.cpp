// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <utils/common.hpp>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include "mri_reconstruction_demo.hpp"
#include "npy_reader.hpp"

bool ParseAndCheckCommandLine(int argc, char *argv[]);

static cv::Mat tensorToMat(const ov::Tensor& tensor);

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
    PerformanceMetrics metrics;

    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    slog::info << ov::get_openvino_version() << slog::endl;
    ov::Core core;

    slog::info << "Reading model " << FLAGS_m << slog::endl;
    std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
    logBasicModelInfo(model);

    std::string outputTensorName = "";
    ov::Layout outputLayout("NHWC");
    for (const auto& output : model->outputs()) {
        if (output.get_shape()[ov::layout::channels_idx(outputLayout)] == 1) {
            outputTensorName = output.get_any_name();
        }
    }
    if (outputTensorName.empty()) {
        throw std::logic_error("Not found suitable output!");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().model().set_layout("NHWC");
    model = ppp.build();

    ov::CompiledModel compiledModel = core.compile_model(model, FLAGS_d);
    logCompiledModelInfo(compiledModel, FLAGS_m, FLAGS_d);

    ov::InferRequest infReq = compiledModel.create_infer_request();

    // Hybrid-CS-Model-MRI/Data/sampling_mask_20perc.npy
    MRIData mri;
    mri.samplingMask = blobFromNPY(FLAGS_p);
    slog::info << "Sampling ratio: " << 1.0 - cv::mean(mri.samplingMask)[0] << slog::endl;

    mri.data = blobFromNPY(FLAGS_i);
    CV_Assert(mri.data.depth() == CV_64F);
    const int numSlices = mri.data.size[0];
    const int height = mri.data.size[1];
    const int width = mri.data.size[2];
    mri.data /= sqrt(height * width);

    mri.reconstructed.create({ numSlices, height, width }, CV_8U);

    slog::info << "Compute..." << slog::endl;

    cv::Mat inputBlob = tensorToMat(infReq.get_input_tensor());
    cv::Mat outputBlob = tensorToMat(infReq.get_tensor(outputTensorName));
    outputBlob = outputBlob.reshape(1, height);

    const auto startTime = std::chrono::steady_clock::now();
    for (int i = 0; i < numSlices; ++i) {
        // Prepare input
        cv::Mat kspace = cv::Mat(height, width, CV_64FC2, mri.data.ptr<double>(i)).clone();

        kspace.setTo(0, mri.samplingMask);
        // TODO: merge the two following lines after OpenCV3 is droppped
        kspace -= cv::Scalar(mri.stats[0], mri.stats[0]);
        kspace /= cv::Mat{cv::Scalar(mri.stats[1], mri.stats[1])};
        kspace.reshape(1, 1).convertTo(inputBlob.reshape(1, 1), CV_32F);
        // Forward pass
        infReq.infer();

        // Save prediction
        cv::Mat slice(height, width, CV_8UC1, mri.reconstructed.ptr<uint8_t>(i));
        cv::normalize(outputBlob, slice, 255, 0, cv::NORM_MINMAX, CV_8U);
    }

    metrics.update(startTime);
    slog::info << "Metrics report:" << slog::endl;
    slog::info << "\tLatency: " << std::fixed << std::setprecision(1) << metrics.getTotal().latency << " ms" << slog::endl;

    // Visualization loop.
    if (!FLAGS_no_show) {
        int sliceId = numSlices / 2;
        cv::namedWindow(kWinName, cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("Slice", kWinName, nullptr, numSlices - 1, callback, &mri);
        callback(sliceId, &mri);  // Trigger initial visualization
        cv::waitKey();
    }

    return 0;
}

cv::Mat tensorToMat(const ov::Tensor& tensor) {
    // NOTE: OpenVINO runtime sizes are reversed.
    ov::Shape tensorShape = tensor.get_shape();
    std::vector<int> size;
    std::transform(tensorShape.begin(), tensorShape.end(), std::back_inserter(size), [](size_t dim) -> int { return int(dim); });
    ov::element::Type precision = tensor.get_element_type();
    CV_Assert(precision == ov::element::f32);
    return cv::Mat(size, CV_32F, (void*)tensor.data());
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
    cv::putText(render, cv::format("Sampled (PSNR %.1f)", cv::PSNR(img, masked)),
                cv::Point(width, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, 0);
    cv::putText(render, cv::format("Reconstructed (PSNR %.1f)", cv::PSNR(img, rec)),
                cv::Point(width*2, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, 0);

    cv::imshow(kWinName, render);
    cv::waitKey(1);
}

cv::Mat kspaceToImage(const cv::Mat& kspace) {
    assert(kspace.dims == 2);
    assert(kspace.channels() == 2);

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
