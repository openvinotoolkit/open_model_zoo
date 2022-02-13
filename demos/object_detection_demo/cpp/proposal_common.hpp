// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <models/model_base.h>
#include <models/results.h>
#include <pipelines/metadata.h>  // TODO: remove
#include <utils/config_factory.h>
#include <utils/performance_metrics.hpp>
#include <utils/slog.hpp>

#include <chrono>
#include <forward_list>
#include <queue>
#include <vector>

using Processor = ModelBase;

class ColorPalette {
private:
    std::vector<cv::Scalar> palette;

    static double getRandom(double a = 0.0, double b = 1.0) {
        static std::default_random_engine e;
        std::uniform_real_distribution<> dis(a, std::nextafter(b, std::numeric_limits<double>::max()));
        return dis(e);
    }

    static double distance(const cv::Scalar& c1, const cv::Scalar& c2) {
        auto dh = std::fmin(std::fabs(c1[0] - c2[0]), 1 - fabs(c1[0] - c2[0])) * 2;
        auto ds = std::fabs(c1[1] - c2[1]);
        auto dv = std::fabs(c1[2] - c2[2]);

        return dh * dh + ds * ds + dv * dv;
    }

    static cv::Scalar maxMinDistance(const std::vector<cv::Scalar>& colorSet, const std::vector<cv::Scalar>& colorCandidates) {
        std::vector<double> distances;
        distances.reserve(colorCandidates.size());
        for (auto& c1 : colorCandidates) {
            auto min = *std::min_element(colorSet.begin(), colorSet.end(),
                [&c1](const cv::Scalar& a, const cv::Scalar& b) { return distance(c1, a) < distance(c1, b); });
            distances.push_back(distance(c1, min));
        }
        auto max = std::max_element(distances.begin(), distances.end());
        return colorCandidates[std::distance(distances.begin(), max)];
    }

    static cv::Scalar hsv2rgb(const cv::Scalar& hsvColor) {
        cv::Mat rgb;
        cv::Mat hsv(1, 1, CV_8UC3, hsvColor);
        cv::cvtColor(hsv, rgb, cv::COLOR_HSV2RGB);
        return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
    }

public:
    explicit ColorPalette(size_t n) {
        palette.reserve(n);
        std::vector<cv::Scalar> hsvColors(1, { 1., 1., 1. });
        std::vector<cv::Scalar> colorCandidates;
        size_t numCandidates = 100;

        hsvColors.reserve(n);
        colorCandidates.resize(numCandidates);
        for (size_t i = 1; i < n; ++i) {
            std::generate(colorCandidates.begin(), colorCandidates.end(),
                []() { return cv::Scalar{ getRandom(), getRandom(0.8, 1.0), getRandom(0.5, 1.0) }; });
            hsvColors.push_back(maxMinDistance(hsvColors, colorCandidates));
        }

        for (auto& hsv : hsvColors) {
            // Convert to OpenCV HSV format
            hsv[0] *= 179;
            hsv[1] *= 255;
            hsv[2] *= 255;

            palette.push_back(hsv2rgb(hsv));
        }
    }

    const cv::Scalar& operator[] (size_t index) const {
        return palette[index % palette.size()];
    }
};

struct LockedExceptions {
    std::forward_list<std::exception_ptr> exceptions;
    std::mutex mtx;

    void push_front(std::exception_ptr exception) {
        const std::lock_guard<std::mutex> lock{mtx};
        exceptions.push_front(exception);
    }
    bool empty() {  // TODO: need atomic?
        const std::lock_guard<std::mutex> lock{mtx};
        return exceptions.empty();
    }
    void may_rethrow() {
        if (!empty()) {
            const std::lock_guard<std::mutex> lock{mtx};
            std::exception_ptr oldest;
            for (std::exception_ptr excpetion : exceptions) {
                oldest = excpetion;
            }
            std::rethrow_exception(oldest);
        }
    }
};

template<typename Meta>
struct OvInferrer {
    struct State {
        State(ov::InferRequest&& ireq): ireq{std::move(ireq)} {}
        ov::InferRequest ireq;
        virtual Meta* data() {return nullptr;}
    };
    struct WithData: State {
        WithData(ov::InferRequest&& ireq, Meta&& meta): State{std::move(ireq)}, meta{std::move(meta)} {}
        Meta meta;
        Meta* data() override {return &meta;}
    };
    struct Iterate {
        OvInferrer& inferrer;
        struct InputIt {
            OvInferrer& inferrer;

            InputIt(OvInferrer& inferrer) : inferrer{inferrer} {}

            bool operator!=(InputIt) const noexcept {return !inferrer.stop_submit || !inferrer.busy_ireqs.empty();};
            InputIt& operator++() {return *this;}
            std::unique_ptr<State> operator*() {return inferrer.state();}
        };
        InputIt begin() {return inferrer;}
        InputIt end() {return inferrer;}
    };

    std::vector<ov::InferRequest> empty_ireqs;
    std::deque<std::pair<ov::InferRequest, Meta>> busy_ireqs;
    bool stop_submit;

    OvInferrer(std::vector<ov::InferRequest> ireqs) : stop_submit{false} {
        for (auto ireq: ireqs) {
            empty_ireqs.push_back(ireq);
        }
    }

    ~OvInferrer() {
        for (auto& ireq : busy_ireqs) {
            ireq.first.cancel();
        }
    }

    void end() {
        if (stop_submit) {
            throw runtime_error("Input was over. Unexpected end");
        }
        stop_submit = true;
    }

    void submit(ov::InferRequest&& ireq, Meta&& meta) {
        if (stop_submit) {
            throw runtime_error("Input was over. Unexpected submit");
        }
        ireq.start_async();
        busy_ireqs.emplace_back(std::move(ireq), std::move(meta));
    }

    std::unique_ptr<State> state() {
        if (!busy_ireqs.empty() && busy_ireqs.front().first.wait_for(std::chrono::milliseconds{0})) {
            auto pair = std::move(busy_ireqs.front());
            busy_ireqs.pop_front();
            empty_ireqs.push_back(pair.first);
            return std::unique_ptr<WithData>(new WithData{std::move(pair.first), std::move(pair.second)});
        }
        if (!(stop_submit || empty_ireqs.empty())) {
            std::unique_ptr<State> res{new State{std::move(empty_ireqs.back())}};
            empty_ireqs.pop_back();
            return res;
        }
        auto pair = std::move(busy_ireqs.front());
        busy_ireqs.pop_front();
        empty_ireqs.push_back(pair.first);
        pair.first.wait();
        return std::unique_ptr<WithData>(new WithData{std::move(pair.first), std::move(pair.second)});
    }

    PerformanceMetrics getInferenceMetircs() {return {};}
};
