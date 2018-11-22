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
    using DrawFunc = std::function<bool(const std::vector<std::shared_ptr<VideoFrame>>&)>;

    AsyncOutput(bool collectStats, size_t queueSize, DrawFunc drawFunc);
    ~AsyncOutput();
    void push(std::vector<std::shared_ptr<VideoFrame>>&& item);
    void start();
    bool isAlive() const;
    struct Stats {
        float renderTime;
    };
    Stats getStats() const;

private:
    const size_t queueSize;
    DrawFunc drawFunc;
    std::queue<std::vector<std::shared_ptr<VideoFrame>>> queue;
    std::atomic_bool terminate = {false};
    std::thread thread;
    std::mutex mutex;
    std::condition_variable condVar;

    PerfTimer perfTimer;
};
