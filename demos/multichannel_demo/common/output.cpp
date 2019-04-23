// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
