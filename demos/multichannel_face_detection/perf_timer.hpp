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

#include <cassert>
#include <vector>
#include <chrono>
#include <atomic>
#include <numeric>

class PerfTimer final {
    const size_t maxCount;
    using duration = std::chrono::duration<float, std::milli>;
    std::vector<duration> values;
    std::atomic<float> avgValue = {0.0f};

public:
    enum {
        DefaultIterationsCount = 50
    };

    explicit PerfTimer(size_t maxCount_);

    template<typename T>
    void addValue(const T& dur) {
        assert(enabled());
        values.push_back(std::chrono::duration_cast<duration>(dur));
        if (values.size() >= maxCount) {
            auto res = std::accumulate(values.begin(),
                                       values.end(),
                                       duration(0.0f));
            avgValue = res.count() / static_cast<float>(values.size());
            values.clear();
        }
    }

    float getValue() const;

    bool enabled() const;
};

struct ScopedTimer final{
    PerfTimer& timer;
    std::chrono::high_resolution_clock::time_point start;

    explicit ScopedTimer(PerfTimer& t):
        timer(t),
        start(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer(){
        auto end = std::chrono::high_resolution_clock::now();
        timer.addValue(end - start);
    }
};
