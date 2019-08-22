// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include <samples/common.hpp>

#include "human_pose_estimator.hpp"
#include "peak.hpp"

namespace human_pose_estimation {
const size_t HumanPoseEstimator::keypointsNumber = 18;

HumanPoseEstimator::HumanPoseEstimator(const std::string& modelPath,
                                       const std::string& targetDeviceName_,
                                       bool enablePerformanceReport)
    : minJointsNumber(3),
      stride(8),
      pad(cv::Vec4i::all(0)),
      meanPixel(cv::Vec3f::all(128)),
      minPeaksDistance(3.0f),
      midPointsScoreThreshold(0.05f),
      foundMidPointsRatioThreshold(0.8f),
      minSubsetScore(0.2f),
      inputLayerSize(-1, -1),
      upsampleRatio(4),
      targetDeviceName(targetDeviceName_),
      enablePerformanceReport(enablePerformanceReport),
      modelPath(modelPath) {
    if (enablePerformanceReport) {
        ie.SetConfig({{InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
                       InferenceEngine::PluginConfigParams::YES}});
    }
    netReader.ReadNetwork(modelPath);
    std::string binFileName = fileNameNoExt(modelPath) + ".bin";
    netReader.ReadWeights(binFileName);
    network = netReader.getNetwork();
    InferenceEngine::InputInfo::Ptr inputInfo = network.getInputsInfo().begin()->second;
    inputLayerSize = cv::Size(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]);
    inputInfo->setPrecision(InferenceEngine::Precision::U8);

    InferenceEngine::OutputsDataMap outputInfo = network.getOutputsInfo();
    auto outputBlobsIt = outputInfo.begin();
    pafsBlobName = outputBlobsIt->first;
    heatmapsBlobName = (++outputBlobsIt)->first;

    executableNetwork = ie.LoadNetwork(network, targetDeviceName);
    request = executableNetwork.CreateInferRequest();
}

std::vector<HumanPose> HumanPoseEstimator::estimate(const cv::Mat& image) {
    CV_Assert(image.type() == CV_8UC3);

    cv::Size imageSize = image.size();
    if (inputWidthIsChanged(imageSize)) {
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        InferenceEngine::SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[2] = inputLayerSize.height;
        input_shape[3] = inputLayerSize.width;
        input_shapes[input_name] = input_shape;
        network.reshape(input_shapes);
        executableNetwork = ie.LoadNetwork(network, targetDeviceName);
        request = executableNetwork.CreateInferRequest();
    }
    InferenceEngine::Blob::Ptr input = request.GetBlob(network.getInputsInfo().begin()->first);
    auto buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type *>();
    preprocess(image, buffer);

    request.Infer();

    InferenceEngine::Blob::Ptr pafsBlob = request.GetBlob(pafsBlobName);
    InferenceEngine::Blob::Ptr heatMapsBlob = request.GetBlob(heatmapsBlobName);
    CV_Assert(heatMapsBlob->getTensorDesc().getDims()[1] == keypointsNumber + 1);
    InferenceEngine::SizeVector heatMapDims =
            heatMapsBlob->getTensorDesc().getDims();
    std::vector<HumanPose> poses = postprocess(
            heatMapsBlob->buffer(),
            heatMapDims[2] * heatMapDims[3],
            keypointsNumber,
            pafsBlob->buffer(),
            heatMapDims[2] * heatMapDims[3],
            pafsBlob->getTensorDesc().getDims()[1],
            heatMapDims[3], heatMapDims[2], imageSize);

    return poses;
}

void HumanPoseEstimator::preprocess(const cv::Mat& image, uint8_t* buffer) const {
    cv::Mat resizedImage;
    double scale = inputLayerSize.height / static_cast<double>(image.rows);
    cv::resize(image, resizedImage, cv::Size(), scale, scale, cv::INTER_CUBIC);
    cv::Mat paddedImage;
    cv::copyMakeBorder(resizedImage, paddedImage, pad(0), pad(2), pad(1), pad(3),
                       cv::BORDER_CONSTANT, meanPixel);
    std::vector<cv::Mat> planes(3);
    for (size_t pId = 0; pId < planes.size(); pId++) {
        planes[pId] = cv::Mat(inputLayerSize, CV_8UC1,
                              buffer + pId * inputLayerSize.area());
    }
    cv::split(paddedImage, planes);
}

std::vector<HumanPose> HumanPoseEstimator::postprocess(
        const float* heatMapsData, const int heatMapOffset, const int nHeatMaps,
        const float* pafsData, const int pafOffset, const int nPafs,
        const int featureMapWidth, const int featureMapHeight,
        const cv::Size& imageSize) const {
    std::vector<cv::Mat> heatMaps(nHeatMaps);
    for (size_t i = 0; i < heatMaps.size(); i++) {
        heatMaps[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                              reinterpret_cast<void*>(
                                  const_cast<float*>(
                                      heatMapsData + i * heatMapOffset)));
    }
    resizeFeatureMaps(heatMaps);

    std::vector<cv::Mat> pafs(nPafs);
    for (size_t i = 0; i < pafs.size(); i++) {
        pafs[i] = cv::Mat(featureMapHeight, featureMapWidth, CV_32FC1,
                          reinterpret_cast<void*>(
                              const_cast<float*>(
                                  pafsData + i * pafOffset)));
    }
    resizeFeatureMaps(pafs);

    std::vector<HumanPose> poses = extractPoses(heatMaps, pafs);
    correctCoordinates(poses, heatMaps[0].size(), imageSize);
    return poses;
}

class FindPeaksBody: public cv::ParallelLoopBody {
public:
    FindPeaksBody(const std::vector<cv::Mat>& heatMaps, float minPeaksDistance,
                  std::vector<std::vector<Peak> >& peaksFromHeatMap)
        : heatMaps(heatMaps),
          minPeaksDistance(minPeaksDistance),
          peaksFromHeatMap(peaksFromHeatMap) {}

    virtual void operator()(const cv::Range& range) const {
        for (int i = range.start; i < range.end; i++) {
            findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i);
        }
    }

private:
    const std::vector<cv::Mat>& heatMaps;
    float minPeaksDistance;
    std::vector<std::vector<Peak> >& peaksFromHeatMap;
};

std::vector<HumanPose> HumanPoseEstimator::extractPoses(
        const std::vector<cv::Mat>& heatMaps,
        const std::vector<cv::Mat>& pafs) const {
    std::vector<std::vector<Peak> > peaksFromHeatMap(heatMaps.size());
    FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())),
                      findPeaksBody);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto& peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }
    std::vector<HumanPose> poses = groupPeaksToPoses(
                peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
                foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);
    return poses;
}

void HumanPoseEstimator::resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const {
    for (auto& featureMap : featureMaps) {
        cv::resize(featureMap, featureMap, cv::Size(),
                   upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
    }
}

void HumanPoseEstimator::correctCoordinates(std::vector<HumanPose>& poses,
                                            const cv::Size& featureMapsSize,
                                            const cv::Size& imageSize) const {
    CV_Assert(stride % upsampleRatio == 0);

    cv::Size fullFeatureMapSize = featureMapsSize * stride / upsampleRatio;

    float scaleX = imageSize.width /
            static_cast<float>(fullFeatureMapSize.width - pad(1) - pad(3));
    float scaleY = imageSize.height /
            static_cast<float>(fullFeatureMapSize.height - pad(0) - pad(2));
    for (auto& pose : poses) {
        for (auto& keypoint : pose.keypoints) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x *= stride / upsampleRatio;
                keypoint.x -= pad(1);
                keypoint.x *= scaleX;

                keypoint.y *= stride / upsampleRatio;
                keypoint.y -= pad(0);
                keypoint.y *= scaleY;
            }
        }
    }
}

bool HumanPoseEstimator::inputWidthIsChanged(const cv::Size& imageSize) {
    double scale = static_cast<double>(inputLayerSize.height) / static_cast<double>(imageSize.height);
    cv::Size scaledSize(static_cast<int>(cvRound(imageSize.width * scale)),
                        static_cast<int>(cvRound(imageSize.height * scale)));
    cv::Size scaledImageSize(std::max(scaledSize.width, inputLayerSize.height),
                             inputLayerSize.height);
    int minHeight = std::min(scaledImageSize.height, scaledSize.height);
    scaledImageSize.width = static_cast<int>(std::ceil(
                scaledImageSize.width / static_cast<float>(stride))) * stride;
    pad(0) = static_cast<int>(std::floor((scaledImageSize.height - minHeight) / 2.0));
    pad(1) = static_cast<int>(std::floor((scaledImageSize.width - scaledSize.width) / 2.0));
    pad(2) = scaledImageSize.height - minHeight - pad(0);
    pad(3) = scaledImageSize.width - scaledSize.width - pad(1);
    if (scaledSize.width == (inputLayerSize.width - pad(1) - pad(3))) {
        return false;
    }

    inputLayerSize.width = scaledImageSize.width;
    return true;
}

HumanPoseEstimator::~HumanPoseEstimator() {
    try {
        if (enablePerformanceReport) {
            std::cout << "Performance counts for " << modelPath << std::endl << std::endl;
            printPerformanceCounts(request, std::cout, getFullDeviceName(ie, targetDeviceName), false);
        }
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
    }
}
}  // namespace human_pose_estimation
