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


AsyncOutput::AsyncOutput(bool collect_stats_, size_t queue_size_,
                         DrawFunc draw_func_):
    queue_size(queue_size_),
    draw_func(std::move(draw_func_)),
    perf_timer(collect_stats_ ? 50 : 0) {}

AsyncOutput::~AsyncOutput() {
    terminate = true;
    cond_var.notify_one();
    if (thread.joinable()) {
        thread.join();
    }
}

void AsyncOutput::push(std::vector<std::shared_ptr<VideoFrame> > &&item) {
    std::unique_lock<std::mutex> lock(mutex);
    while (queue.size() >= queue_size) {
        queue.pop();
    }
    queue.push(std::move(item));
    lock.unlock();
    cond_var.notify_one();
}

void AsyncOutput::start() {
    thread = std::thread([&]() {
        std::vector<std::shared_ptr<VideoFrame>> elem;
        while (!terminate) {
            std::unique_lock<std::mutex> lock(mutex);
            cond_var.wait(lock, [&]() {
                return !queue.empty() || terminate;
            });
            if (terminate) {
                break;
            }

            elem = std::move(queue.front());
            queue.pop();
            lock.unlock();

            if (perf_timer.enabled()) {
                ScopedTimer sc(perf_timer);
                if (!draw_func(elem)) {
                    terminate = true;
                }
            } else {
                if (!draw_func(elem)) {
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
    return Stats{perf_timer.getValue()};
}
