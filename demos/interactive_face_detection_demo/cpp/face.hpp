// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

# pragma once
#include <string>
#include <map>
#include <memory>
#include <utility>
#include <list>
#include <vector>
#include <opencv2/opencv.hpp>

#include "detectors.hpp"

// -------------------------Describe detected face on a frame-------------------------------------------------

struct Face {
public:
    using Ptr = std::shared_ptr<Face>;

    explicit Face(size_t id, cv::Rect& location);

    void updateAge(float value);
    void updateGender(float value);
    void updateEmotions(std::map<std::string, float> values);
    void updateHeadPose(HeadPoseDetection::Results values);
    void updateLandmarks(std::vector<float> values);
    void updateRealFaceConfidence(float value);

    int getAge();
    bool isMale();
    bool isReal();
    std::map<std::string, float> getEmotions();
    std::pair<std::string, float> getMainEmotion();
    HeadPoseDetection::Results getHeadPose();
    const std::vector<float>& getLandmarks();
    size_t getId();

    void ageGenderEnable(bool value);
    void emotionsEnable(bool value);
    void headPoseEnable(bool value);
    void landmarksEnable(bool value);
    void antispoofingEnable(bool value);

    bool isAgeGenderEnabled();
    bool isEmotionsEnabled();
    bool isHeadPoseEnabled();
    bool isLandmarksEnabled();
    bool isAntispoofingEnabled();

public:
    cv::Rect _location;
    float _intensity_mean;

private:
    size_t _id;
    float _age;
    float _maleScore;
    float _femaleScore;
    std::map<std::string, float> _emotions;
    HeadPoseDetection::Results _headPose;
    std::vector<float> _landmarks;
    float _realFaceConfidence;

    bool _isAgeGenderEnabled;
    bool _isEmotionsEnabled;
    bool _isHeadPoseEnabled;
    bool _isLandmarksEnabled;
    bool _isAntispoofingEnabled;
};

// ----------------------------------- Utils -----------------------------------------------------------------
float calcIoU(cv::Rect& src, cv::Rect& dst);
float calcMean(const cv::Mat& src);
Face::Ptr matchFace(cv::Rect rect, std::list<Face::Ptr>& faces);
