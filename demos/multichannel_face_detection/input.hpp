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

#include "graph.hpp"

struct Face;
class VideoFrame{
public:
    std::shared_ptr<cv::Mat> frame;
    std::size_t source_idx;
    std::size_t frame_idx;
    std::vector<Face> detections;
    VideoFrame() { frame.reset(new cv::Mat()); frame_idx = 0; source_idx = 0;}

    void operator =(VideoFrame const& vf);
};

class VideoSource;

class VideoSources{
private:
    std::vector<std::unique_ptr<VideoSource>> inputs;
    std::size_t internal_idx;
    const bool isAsync = false;
    const bool collect_stats = false;
    std::vector<size_t> connected_idxs;

    std::thread workThread;
    std::atomic_bool terminate = {true};

    std::size_t duplicate_channels_num = 0;
    bool real_fps;

    const size_t queue_size = 1;
    const size_t polling_time_msec = 1000;

public:
    VideoSources(bool async, std::size_t dc_num, bool collect_stats_,
                 size_t queue_size_, size_t polling_time_msec_, bool real_fps);
    ~VideoSources();

    void stop();

    size_t getNewIdx();

    size_t getInputsNum();

    bool openVideo(const std::string& source);

    void start();

    bool getFrame(VideoFrame& frame);
    bool getFrame(cv::Mat& frame);

    struct Stats {
        std::vector<float> read_times;
    };

    Stats getStats() const;
};
