// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "face.hpp"

#include <algorithm>
#include <cmath>

Face::Face(size_t id, cv::Rect& location)
    : _location(location),
      _intensity_mean(0.f),
      _id(id),
      _age(-1),
      _maleScore(0),
      _femaleScore(0),
      _yaw(0.f),
      _pitch(0.f),
      _roll(0.f),
      _realFaceConfidence(0.f) {}

void Face::updateAge(float value) {
    _age = (_age == -1) ? value : 0.95f * _age + 0.05f * value;
}

void Face::updateGender(float value) {
    if (value < 0)
        return;

    if (value > 0.5) {
        _maleScore += value - 0.5f;
    } else {
        _femaleScore += 0.5f - value;
    }
}

void Face::updateEmotions(const std::map<std::string, float>& values) {
    for (auto& kv : values) {
        if (_emotions.find(kv.first) == _emotions.end()) {
            _emotions[kv.first] = kv.second;
        } else {
            _emotions[kv.first] = 0.9f * _emotions[kv.first] + 0.1f * kv.second;
        }
    }
}

void Face::updateHeadPose(float y, float p, float r) {
    _yaw = y;
    _pitch = p;
    _roll = r;
}

void Face::updateLandmarks(std::vector<float> values) {
    _landmarks = std::move(values);
}

void Face::updateRealFaceConfidence(float value) {
    _realFaceConfidence = value;
}

int Face::getAge() {
    return static_cast<int>(std::floor(_age + 0.5f));
}

bool Face::isMale() {
    return _maleScore > _femaleScore;
}

bool Face::isReal() {
    return _realFaceConfidence > 50.f;
}

std::map<std::string, float> Face::getEmotions() {
    return _emotions;
}

std::pair<std::string, float> Face::getMainEmotion() {
    auto x = std::max_element(_emotions.begin(),
                              _emotions.end(),
                              [](const std::pair<std::string, float>& p1, const std::pair<std::string, float>& p2) {
                                  return p1.second < p2.second;
                              });

    return *x;
}

const std::vector<float>& Face::getLandmarks() {
    return _landmarks;
}

size_t Face::getId() {
    return _id;
}

float calcIoU(cv::Rect& src, cv::Rect& dst) {
    cv::Rect i = src & dst;
    cv::Rect u = src | dst;

    return static_cast<float>(i.area()) / static_cast<float>(u.area());
}

Face::Ptr matchFace(cv::Rect rect, const std::list<Face::Ptr>& faces) {
    Face::Ptr face(nullptr);
    float maxIoU = 0.55f;

    for (auto&& f : faces) {
        float iou = calcIoU(rect, f->_location);
        if (iou > maxIoU) {
            face = f;
            maxIoU = iou;
        }
    }

    return face;
}
