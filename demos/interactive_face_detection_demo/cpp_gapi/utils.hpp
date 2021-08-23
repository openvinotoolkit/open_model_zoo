// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

# pragma once

#include <chrono>
#include <map>
#include <stdexcept>
#include <string>

class CallStat {
public:
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    CallStat():
        _number_of_calls(0), _total_duration(0.0), _last_call_duration(0.0), _smoothed_duration(-1.0) {
    }

    double getSmoothedDuration() {
        // Additional check is needed for the first frame while duration of the first
        // visualisation is not calculated yet.
        if (_smoothed_duration < 0) {
            auto t = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<ms>(t - _last_call_start).count();
        }
        return _smoothed_duration;
    }

    double getTotalDuration() {
        return _total_duration;
    }

    double getLastCallDuration() {
        return _last_call_duration;
    }

    void calculateDuration() {
        auto t = std::chrono::steady_clock::now();
        _last_call_duration = std::chrono::duration_cast<ms>(t - _last_call_start).count();
        _number_of_calls++;
        _total_duration += _last_call_duration;
        if (_smoothed_duration < 0) {
            _smoothed_duration = _last_call_duration;
        }
        double alpha = 0.1;
        _smoothed_duration = _smoothed_duration * (1.0 - alpha) + _last_call_duration * alpha;
        _last_call_start = t;
    }

    void setStartTime() {
        _last_call_start = std::chrono::steady_clock::now();
    }

private:
    size_t _number_of_calls;
    double _total_duration;
    double _last_call_duration;
    double _smoothed_duration;
    std::chrono::time_point<std::chrono::steady_clock> _last_call_start;
};

class Timer {
public:
    void start(const std::string& name) {
        _timers[name].setStartTime();
    }

    void finish(const std::string& name) {
        auto& timer = (*this)[name];
        timer.calculateDuration();
    }

    CallStat& operator[](const std::string& name) {
        if (_timers.find(name) == _timers.end()) {
            throw std::logic_error("No timer with name " + name + ".");
        }
        return _timers[name];
    }

private:
    std::map<std::string, CallStat> _timers;
};
