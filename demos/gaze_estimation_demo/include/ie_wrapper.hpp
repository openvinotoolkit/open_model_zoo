// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <string>

#include <map>
#include <vector>

#include <inference_engine.hpp>


#include <ie_iextension.h>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <samples/common.hpp>

namespace gaze_estimation {
class IEWrapper {
public:
    IEWrapper(InferenceEngine::Core& ie,
              const std::string& modelPath,
              const std::string& deviceName);
    // For setting input blobs containing images
    void setInputBlob(const std::string& blobName, const cv::Mat& image);
    // For setting input blobs containing vectors of data
    void setInputBlob(const std::string& blobName, const std::vector<float>& data);

    // Get output blob content as a vector given its name
    void getOutputBlob(const std::string& blobName, std::vector<float>& output);

    void printPerlayerPerformance() const;

    const std::map<std::string, std::vector<unsigned long>>& getInputBlobDimsInfo() const;
    const std::map<std::string, std::vector<unsigned long>>& getOutputBlobDimsInfo() const;

    std::string expectSingleInput() const;
    std::string expectSingleOutput() const;

    void expectImageInput(const std::string& blobName) const;

    void reshape(const std::map<std::string, std::vector<unsigned long>>& newBlobsDimsInfo);

    void infer();

private:
    std::string modelPath;
    std::string deviceName;
    InferenceEngine::Core& ie;
    InferenceEngine::CNNNetwork network;
    InferenceEngine::ExecutableNetwork executableNetwork;
    InferenceEngine::InferRequest request;
    std::map<std::string, std::vector<unsigned long>> inputBlobsDimsInfo;
    std::map<std::string, std::vector<unsigned long>> outputBlobsDimsInfo;

    void setExecPart();
};
}  // namespace gaze_estimation
