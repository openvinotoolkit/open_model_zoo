// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
