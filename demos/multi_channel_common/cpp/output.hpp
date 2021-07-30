// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <memory>

#include "graph.hpp"
#include "perf_timer.hpp"

class AsyncOutput{
public:
    using DrawFunc = std::function<bool(const VideoFrame::FramesWithTimeStamp&)>;

    AsyncOutput(bool collectStats, size_t queueSize, DrawFunc drawFunc);
    ~AsyncOutput();
    void push(VideoFrame::FramesWithTimeStamp&& item);
    void start();
    bool isAlive() const;
    struct Stats {
        float renderTime;
    };
    Stats getStats() const;

private:
    const size_t queueSize;
    DrawFunc drawFunc;
    std::queue<VideoFrame::FramesWithTimeStamp> queue;
    std::atomic_bool terminate = {false};
    std::thread thread;
    std::mutex mutex;
    std::condition_variable condVar;

    PerfTimer perfTimer;
};
