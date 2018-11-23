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
#include <memory>
#include <vector>
#include <utility>

#include "output.hpp"

AsyncOutput::AsyncOutput(bool collectStats, size_t queueSize,
                         DrawFunc drawFunc):
    queueSize(queueSize),
    drawFunc(std::move(drawFunc)),
    perfTimer(collectStats ? PerfTimer::DefaultIterationsCount : 0) {}

AsyncOutput::~AsyncOutput() {
    terminate = true;
    condVar.notify_one();
    if (thread.joinable()) {
        thread.join();
    }
}

void AsyncOutput::push(std::vector<std::shared_ptr<VideoFrame> > &&item) {
    std::unique_lock<std::mutex> lock(mutex);
    while (queue.size() >= queueSize) {
        queue.pop();
    }
    queue.push(std::move(item));
    lock.unlock();
    condVar.notify_one();
}

void AsyncOutput::start() {
    thread = std::thread([&]() {
        std::vector<std::shared_ptr<VideoFrame>> elem;
        while (!terminate) {
            std::unique_lock<std::mutex> lock(mutex);
            condVar.wait(lock, [&]() {
                return !queue.empty() || terminate;
            });
            if (terminate) {
                break;
            }

            elem = std::move(queue.front());
            queue.pop();
            lock.unlock();

            if (perfTimer.enabled()) {
                ScopedTimer sc(perfTimer);
                if (!drawFunc(elem)) {
                    terminate = true;
                }
            } else {
                if (!drawFunc(elem)) {
                    terminate = true;
                }
            }
        }
    });
}


bool AsyncOutput::isAlive() const {
    return !terminate;
}

AsyncOutput::Stats AsyncOutput::getStats() const {
    return Stats{perfTimer.getValue()};
}
