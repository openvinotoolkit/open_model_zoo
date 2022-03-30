// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>  // for size_t

#include <list>  // for list
#include <map>  // for map
#include <memory>  // for shared_ptr
#include <string>  // for string
#include <utility>  // for pair
#include <vector>  // for vector

#include <opencv2/core.hpp>  // for Rect

// -------------------------Describe detected face on a frame-------------------

struct Face {
public:
    using Ptr = std::shared_ptr<Face>;

    explicit Face(size_t id, cv::Rect& location);

    void updateAge(float value);
    void updateGender(float value);
    void updateEmotions(const std::map<std::string, float>& values);
    void updateHeadPose(float y, float p, float r);
    void updateLandmarks(std::vector<float> values);
    void updateRealFaceConfidence(float value);

    int getAge();
    bool isMale();
    bool isReal();
    std::map<std::string, float> getEmotions();
    std::pair<std::string, float> getMainEmotion();
    const std::vector<float>& getLandmarks();
    size_t getId();

public:
    cv::Rect _location;
    float _intensity_mean;

    size_t _id;
    float _age;
    float _maleScore;
    float _femaleScore;
    std::map<std::string, float> _emotions;
    float _yaw;
    float _pitch;
    float _roll;
    float _realFaceConfidence;
    std::vector<float> _landmarks;
};

// ----------------- Utils -----------------
Face::Ptr matchFace(cv::Rect rect, const std::list<Face::Ptr>& faces);
