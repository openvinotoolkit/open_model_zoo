// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <deque>

class PerformanceMetrics {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = Clock::duration;
    using Ms = std::chrono::milliseconds;
    using Sec = std::chrono::seconds;

    PerformanceMetrics(Duration timeWindow = Sec(1));
    void recalculate(TimePoint lastRequestStartTime);
    double getFps() const { return fps; };
    double getLatency() const { return latency; }
    double getTotalFps() const;
    double getTotalLatency() const;
    void stop();
    void reinitialize();
    bool hasStarted() const;

private:
    struct Measurement {
        Duration latency;
        TimePoint timePoint;
    };

    Duration timeWindowSize;
    std::deque<Measurement> measurements;
    double fps;
    double latency;
    Duration latencySum;
    TimePoint startTime;
    TimePoint stopTime;
    unsigned numFramesProcessed;
    Duration latencyTotalSum;
};