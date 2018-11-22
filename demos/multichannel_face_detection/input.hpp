/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
#pragma once

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>

#include <opencv2/opencv.hpp>

#ifdef USE_NATIVE_CAMERA_API
#include "multicam/controller.hpp"
#endif

#include "decoder.hpp"

struct Face {
    cv::Rect2f rect;
    float confidence;
    unsigned char age;
    unsigned char gender;
    Face(cv::Rect2f r, float c, unsigned char a, unsigned char g): rect(r), confidence(c), age(a), gender(g) {}
};


class VideoFrame final {
public:
    cv::Mat frame;
    std::size_t sourceIdx = 0;
    std::vector<Face> detections;
    VideoFrame() = default;

    void operator =(VideoFrame const& vf) = delete;
};

class VideoSource;
class VideoSourceNative;
class VideoSourceOCV;
class VideoSourceStreamFile;

class VideoSources {
private:
    Decoder decoder;
#ifdef USE_NATIVE_CAMERA_API
    mcam::controller controller;
#endif

    std::mutex decode_mutex;  // hardware decoding enqueue lock

    std::vector<std::unique_ptr<VideoSource>> inputs;
    const bool isAsync = false;
    const bool collectStats = false;

    bool realFps;

    const size_t queueSize = 1;
    const size_t pollingTimeMSec = 1000;

    void stop();

    friend VideoSourceNative;
    friend VideoSourceOCV;
    friend VideoSourceStreamFile;

public:
    struct InitParams {
        std::size_t queueSize = 5;
        std::size_t pollingTimeMSec = 1000;
        bool isAsync = true;
        bool collectStats = false;
        bool realFps = false;
        unsigned expectedWidth = 0;
        unsigned expectedHeight = 0;
    };

    explicit VideoSources(const InitParams& p);
    ~VideoSources();

    void openVideo(const std::string& source, bool native);

    void start();

    bool getFrame(size_t index, VideoFrame& frame);

    struct Stats {
        std::vector<float> readTimes;
        float decodingLatency = 0.0f;
    };

    Stats getStats() const;
};
