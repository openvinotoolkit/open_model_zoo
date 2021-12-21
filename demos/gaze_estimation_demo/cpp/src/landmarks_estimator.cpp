// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "landmarks_estimator.hpp"

namespace gaze_estimation {
LandmarksEstimator::LandmarksEstimator(InferenceEngine::Core& ie,
                                       const std::string& modelPath,
                                       const std::string& deviceName):
                    ieWrapper(ie, modelPath, modelType, deviceName) {
    inputBlobName = ieWrapper.expectSingleInput();
    ieWrapper.expectImageInput(inputBlobName);

    const auto& outputInfo = ieWrapper.getOutputBlobDimsInfo();

    outputBlobName = ieWrapper.expectSingleOutput();
    const auto& outputBlobDims = outputInfo.at(outputBlobName);

    if (outputBlobDims.size() != 4 && outputBlobDims.size() != 2) {
        throw std::runtime_error(modelPath + " network output layer should have 2 or 4 dimensions");
    }
}

void LandmarksEstimator::estimate(const cv::Mat& image,
                                  FaceInferenceResults& outputResults) {
    auto faceBoundingBox = outputResults.faceBoundingBox;
    auto faceCrop(cv::Mat(image, faceBoundingBox));

    ieWrapper.setInputBlob(inputBlobName, faceCrop);
    ieWrapper.infer();

    const auto& outputInfo = ieWrapper.getOutputBlobDimsInfo();
    const auto& outputBlobDims = outputInfo.at(outputBlobName);
    if (outputBlobDims.size() == 2) {
        outputResults.faceLandmarks=simplePostprocess(faceBoundingBox, faceCrop);
    }
    else {
        outputResults.faceLandmarks = heatMapPostprocess(faceBoundingBox, faceCrop);
    }
}

std::vector<cv::Point2i> LandmarksEstimator::simplePostprocess(cv::Rect faceBoundingBox, cv::Mat faceCrop) {
    std::vector<float> rawLandmarks;
    ieWrapper.getOutputBlob(outputBlobName, rawLandmarks);
    std::vector<cv::Point2i> normedLandmarks;
    for (unsigned long i = 0; i < rawLandmarks.size() / 2; ++i) {
        int x = static_cast<int>(rawLandmarks[2 * i] * faceCrop.cols + faceBoundingBox.tl().x);
        int y = static_cast<int>(rawLandmarks[2 * i + 1] * faceCrop.rows + faceBoundingBox.tl().y);
        normedLandmarks.push_back(cv::Point2i(x, y));
    }
    return normedLandmarks;
}

std::vector<cv::Point2i> LandmarksEstimator::heatMapPostprocess(cv::Rect faceBoundingBox, cv::Mat faceCrop) {
    std::vector<float> rawLandmarks;
    ieWrapper.getOutputBlob(outputBlobName, rawLandmarks);
    const auto& outputInfo = ieWrapper.getOutputBlobDimsInfo();
    const auto& heatMapsDims = outputInfo.at(outputBlobName);
    numberLandmarks = heatMapsDims[1];
    std::vector<cv::Mat> heatMaps = split(rawLandmarks, heatMapsDims);
    float w = faceBoundingBox.width, h = faceBoundingBox.height;
    cv::Point2f center(faceBoundingBox.tl().x + w * 0.5f, faceBoundingBox.tl().y + h * 0.5f);
    cv::Point2f scale(w / 200.0, h / 200.0);

    std::vector<cv::Point2f> preds = getMaxPreds(heatMaps);

    for (size_t landmarkId = 0; landmarkId < numberLandmarks; landmarkId++) {
        const cv::Mat& heatMap = heatMaps[landmarkId];
        int px = int(preds[landmarkId].x);
        int py = int(preds[landmarkId].y);
        if (1 < px && px < heatMap.cols - 1 && 1 < py && py < heatMap.rows - 1) {
            float diffFirst = heatMap.at<float>(py, px + 1) - heatMap.at<float>(py, px - 1);
            float diffSecond = heatMap.at<float>(py + 1, px) - heatMap.at<float>(py - 1, px);
            preds[landmarkId].x += sign(diffFirst) * 0.25f;
            preds[landmarkId].y += sign(diffSecond) * 0.25f;
        }
    }
    //transform preds
    cv::Mat trans = affineTransform(center, scale, 0, heatMapsDims[2], heatMapsDims[3], cv::Point2f(0., 0.), true);
    std::vector<cv::Point2i> landmarks;
    for (size_t landmarkId = 0; landmarkId < numberLandmarks; landmarkId++) {
        cv::Mat coord(3, 1, CV_32F);
        coord.at<float>(0, 0) = preds[landmarkId].x;
        coord.at<float>(1, 0) = preds[landmarkId].y;
        coord.at<float>(2, 0) = 1;
        cv::Mat point;
        trans.convertTo(trans, CV_32F);
        point = trans * coord;
        int x = static_cast<int>(point.at<float>(0, 0));
        int y = static_cast<int>(point.at<float>(1, 0));
        landmarks.push_back(cv::Point2i(x, y));
    }
    return landmarks;
}

std::vector<cv::Mat> LandmarksEstimator::split(std::vector<float>& data, const std::vector<unsigned long>& shape) {
    std::vector<cv::Mat> flattenData(shape[1]);
    float* output = new float[data.size()];
    for (size_t i = 0; i < data.size(); i++) {
        output[i] = data[i];
    }
    for (size_t i = 0; i < flattenData.size(); i++) {
        flattenData[i] = cv::Mat(shape[2], shape[3], CV_32FC1, output + i * shape[2] * shape[3]);
    }
    return flattenData;
}

std::vector<cv::Point2f> LandmarksEstimator::getMaxPreds(std::vector<cv::Mat> heatMaps) {
    std::vector<cv::Point2f> preds;
    size_t reshapedSize = heatMaps[0].cols * heatMaps[0].rows;
    for (size_t landmarkId = 0; landmarkId < numberLandmarks; landmarkId++) {
        const cv::Mat& heatMap = heatMaps[landmarkId];
        const float* heatMapData = heatMap.ptr<float>();
        std::vector<int> indices(reshapedSize);
        std::iota(std::begin(indices), std::end(indices), 0);
        std::partial_sort(std::begin(indices), std::begin(indices) + numberLandmarks, std::end(indices),
            [heatMapData](int l, int r) { return heatMapData[l] > heatMapData[r]; });
        size_t idx = indices[0];
        float maxVal = heatMapData[idx];
        if (maxVal > 0) {
            preds.push_back(cv::Point2f(static_cast<float>(idx % heatMaps[0].cols), static_cast<float>(idx / heatMaps[0].cols)));
        }
        else {
            preds.push_back(cv::Point2f(-1, -1));
        }
    }
    return preds;
}

int LandmarksEstimator::sign(float number) {
    if (number > 0) {
        return 1;
    }
    else if (number < 0) {
        return -1;
    }
    return 0;
}

cv::Mat LandmarksEstimator::affineTransform(cv::Point2f center, cv::Point2f scale,
    float rot, size_t dst_w, size_t dst_h, cv::Point2f shift, bool inv) {
    cv::Point2f scale_tmp = scale * 200.0;
    const float pi = acos(-1.0f);
    float rot_rad = pi * rot / 180;
    cv::Point2f src_dir = rotatePoint(cv::Point2f(0.f, scale_tmp.y * -0.5f), rot_rad);
    cv::Point2f* src = new cv::Point2f[3];
    src[0] = cv::Point2f(center.x + scale_tmp.x * shift.x, center.y + scale_tmp.y * shift.y);
    src[1] = cv::Point2f(center.x + src_dir.x + scale_tmp.x * shift.x, center.y + src_dir.y + scale_tmp.y * shift.y);
    src[2] = get3rdPoint(src[0], src[1]);
    cv::Point2f dst_dir = cv::Point2f(0.f, dst_w * -0.5f);
    cv::Point2f* dst = new cv::Point2f[3];
    dst[0] = cv::Point2f(dst_w * 0.5f, dst_h * 0.5f);
    dst[1] = dst[0] + dst_dir;
    dst[2] = get3rdPoint(dst[0], dst[1]);
    cv::Mat trans;
    if (inv) {
        trans = cv::getAffineTransform(dst, src);
    }
    else {
        trans = cv::getAffineTransform(src, dst);
    }
    return trans;
}

cv::Point2f LandmarksEstimator::rotatePoint(cv::Point2f pt, float angle_rad) {
    float new_x = pt.x * cos(angle_rad) - pt.y * sin(angle_rad);
    float new_y = pt.x * sin(angle_rad) + pt.y * cos(angle_rad);
    return  cv::Point2f(new_x, new_y);
}

cv::Point2f LandmarksEstimator::get3rdPoint(cv::Point2f a, cv::Point2f b) {
    cv::Point2f direction = a - b;
    return  cv::Point2f(b.x - direction.y, b.y + direction.x);
}

LandmarksEstimator::~LandmarksEstimator() {
}
}  // namespace gaze_estimation
