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

#include "input.hpp"

#include <numeric>
#include <chrono>
#include <utility>
#include <queue>
#include <memory>
#include <string>

#include "perf_timer.hpp"

void VideoFrame::operator =(const VideoFrame& vf) {
    this->frame.reset(new cv::Mat(vf.frame.get()->clone()));
    this->frame_idx = vf.frame_idx;
    this->source_idx = vf.source_idx;
    this->detections = vf.detections;
}

class VideoSource {
    PerfTimer perf_timer;
    std::thread workThread;
    const bool isAsync = false;
    std::atomic_bool terminate = {false};
    std::string video_name;

    std::mutex mutex;
    std::condition_variable cond_var;
    std::condition_variable has_frame;
    std::queue<std::pair<bool, cv::Mat>> queue;

    cv::VideoCapture source;
    std::size_t unic_idx;
    std::size_t frame_idx;

    std::size_t duplicate_idx;
    std::size_t duplicate_channels_num;
    bool real_fps;

    const size_t queue_size = 1;
    const size_t polling_time_msec = 1000;

    bool openDevice();

    template<bool CollectStats>
    bool readFrame(cv::Mat& frame);

    template<bool CollectStats>
    bool readFrameImpl(cv::Mat& frame);

    template<bool CollectStats>
    void startImpl();

public:
    VideoSource(bool async, bool collect_stats_, const std::string& name,
                std::size_t uidx, std::size_t dc_num, size_t queue_size_,
                size_t polling_time_msec_, bool real_fps);

    ~VideoSource();

    void start();

    void stop();

    bool read(cv::Mat& frame);
    bool read(VideoFrame& frame);

    float getAvgReadTime() const {
        return perf_timer.getValue();
    }
};

namespace {
bool isNumeric(const std::string& str) {
    return std::strspn(str.c_str(), "0123456789") == str.length();
}
}  // namespace

bool VideoSource::openDevice() {
    static std::mutex initMutex;  // HACK: opencv camera init is not thread-safe
    std::unique_lock<std::mutex> lock(initMutex);
    if (isNumeric(video_name)) {
        bool res = false;
#ifdef __linux__
        res = source.open("/dev/video" + video_name);
#else
        res = source.open(std::stoi(video_name));
#endif
        if (res) {
            source.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
        }
        return res;
    } else {
        return source.open(video_name);
    }
}

template<bool CollectStats>
bool VideoSource::readFrame(cv::Mat& frame) {
    if (!(source.isOpened() || openDevice())) {
        return false;
    }
    if (!readFrameImpl<CollectStats>(frame)) {
        return openDevice() && readFrameImpl<CollectStats>(frame);
    }
    return true;
}

template<bool CollectStats>
bool VideoSource::readFrameImpl(cv::Mat& frame) {
    if (CollectStats) {
        ScopedTimer st(perf_timer);
        return source.read(frame);
    } else {
        return source.read(frame);
    }
}

VideoSource::VideoSource(bool async, bool collect_stats_,
                         const std::string& name, std::size_t uidx,
                         std::size_t dc_num, size_t queue_size_,
                         size_t polling_time_msec_, bool real_fps):
    perf_timer(collect_stats_ ? 50 : 0),
    isAsync(async), video_name(name),
    unic_idx(uidx),
    frame_idx(0),
    duplicate_idx(0),
    duplicate_channels_num(dc_num),
    real_fps(real_fps),
    queue_size(queue_size_),
    polling_time_msec(polling_time_msec_) {}

VideoSource::~VideoSource() {
    stop();
}

template<bool CollectStats>
void VideoSource::startImpl() {
    if (isAsync) {
        terminate = false;
        workThread = std::thread([&]() {
            while (!terminate) {
                {
                    cv::Mat frame;
                    bool result = false;
                    while (!((result = readFrame<CollectStats>(frame)) || terminate)) {
                        std::unique_lock<std::mutex> lock(mutex);
                        if (queue.empty() || queue.back().first) {
                            queue.push({false, frame});
                            lock.unlock();
                            has_frame.notify_one();
                            lock.lock();
                        }
                        std::chrono::milliseconds timeout(polling_time_msec);
                        cond_var.wait_for(lock,
                                          timeout,
                                          [&]() {
                            return terminate.load();
                        });
                    }

                    std::unique_lock<std::mutex> lock(mutex);
                    cond_var.wait(lock, [&]() {
                        return queue.size() < queue_size || terminate;
                    });

                    queue.push({result, frame});
                }
                has_frame.notify_one();
            }
        });
    } else {
        assert(!"Not implemented");
    }
}

void VideoSource::start() {
    if (perf_timer.enabled()) {
        startImpl<true>();
    } else {
        startImpl<false>();
    }
}

void VideoSource::stop() {
    if (isAsync) {
        terminate = true;
        cond_var.notify_one();
        if (workThread.joinable()) {
            workThread.join();
        }
    }
}

bool VideoSource::read(cv::Mat& frame) {
    if (isAsync) {
        size_t count = 0;
        bool res = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            has_frame.wait(lock, [&]() {
                return !queue.empty() || terminate;
            });
            res = queue.front().first;
            frame = queue.front().second;
            if (real_fps || queue.size() > 1 || queue_size == 1) {
                queue.pop();
            }
            count = queue.size();
            (void)count;
        }
        cond_var.notify_one();
        return res;
    } else {
        return source.read(frame);
    }
}

bool VideoSource::read(VideoFrame& frame) {
    if (duplicate_channels_num) {
        frame.source_idx = unic_idx + duplicate_idx;
        duplicate_idx++;
        if (duplicate_idx > duplicate_channels_num)
            duplicate_idx = 0;
    } else {
        frame.source_idx = unic_idx;
    }
    frame.frame_idx  = frame_idx++;
    return read((*frame.frame.get()));
}

VideoSources::VideoSources(bool async, std::size_t dc_num, bool collect_stats_,
                             size_t queue_size_, size_t polling_time_msec_, bool real_fps):
    isAsync(async),
    collect_stats(collect_stats_),
    duplicate_channels_num(dc_num),
    real_fps(real_fps),
    queue_size(queue_size_),
    polling_time_msec(polling_time_msec_) {
    internal_idx = 0;
    terminate = false;
}

VideoSources::~VideoSources() {
    stop();
}

void VideoSources::stop() {
    terminate = true;
    if (workThread.joinable()) {
        workThread.join();
    }
}

size_t VideoSources::getNewIdx() {
    size_t i = 0;
    while (true) {
        if (std::find(connected_idxs.begin(), connected_idxs.end(), i) ==
                connected_idxs.end()) {
            return i;
        }
        i++;
    }
}

size_t VideoSources::getInputsNum() {
    return inputs.size();
}

bool VideoSources::openVideo(const std::string& source) {
    const auto idx = getNewIdx();
    inputs.emplace_back(std::unique_ptr<VideoSource>(
                            new VideoSource(isAsync, collect_stats, source,
                                            idx*(duplicate_channels_num+1),
                                            duplicate_channels_num,
                                            queue_size, polling_time_msec, real_fps)));
    connected_idxs.push_back(idx);
    return true;
}

void VideoSources::start() {
    for (auto& input : inputs) {
        input->start();
    }
}

bool VideoSources::getFrame(VideoFrame& frame) {
    if (inputs.size() > 0) {
        internal_idx++;
        if (internal_idx >= inputs.size()) {
            internal_idx = 0;
        }

        return inputs[internal_idx]->read(frame);
    } else {
        return false;
    }
}

bool VideoSources::getFrame(cv::Mat& frame) {
    if (inputs.size() > 0) {
        internal_idx++;
        if (internal_idx >= inputs.size()) {
            internal_idx = 0;
        }

        return inputs[internal_idx]->read(frame);
    } else {
        return false;
    }
}

VideoSources::Stats VideoSources::getStats() const {
    Stats ret;
    ret.read_times.reserve(inputs.size());
    for (auto& input : inputs) {
        assert(nullptr != input);
        ret.read_times.push_back(input->getAvgReadTime());
    }
    return ret;
}

