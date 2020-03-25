// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <opencv2/opencv.hpp>

#ifdef USE_TBB
#include "threading.hpp"
#endif

class Decoder final {
public:
    enum class Mode {
        Immediate,
        Async,
        Hw
    };

    struct Settings {
        Mode mode = Mode::Immediate;
        unsigned output_width = 0;
        unsigned output_height = 0;
        unsigned num_buffers = 1;
        bool collect_stats = false;
    };

    explicit Decoder(const Settings& s);
    Decoder(const Decoder&) = delete;
    Decoder& operator =(const Decoder&) = delete;
    ~Decoder();

    struct Stats {
        float decoding_latency = 0.0f;
    };

    Stats getStats() const;

    template<typename F>
    void decode(const void* data, size_t size, unsigned width, unsigned height,
                F&& callback) {
        assert(nullptr != data);
        assert(size > 0);
        assert(width > 0);
        assert(height > 0);

        auto mode = settings.mode;
        if (Mode::Immediate == mode) {
            auto img = cv::imdecode(
            {static_cast<const char*>(data),
             static_cast<int>(size)},
                           cv::IMREAD_COLOR);
            callback(std::move(img));
        } else if (Mode::Async == mode) {
#ifdef USE_TBB
            auto decode = [data, size, c = std::move(callback), this]() mutable {
                auto img = cv::imdecode(
                {static_cast<const char*>(data),
                 static_cast<int>(size)},
                            cv::IMREAD_COLOR);
                c(std::move(img));
            };
            auto& arena = get_tbb_arena();
            arena.enqueue(std::move(decode));
#else
            throw std::logic_error("Async decoding is not supported");
#endif
        } else if (Mode::Hw == mode) {
#ifdef USE_LIBVA
            auto decode = [data, size, c = std::move(callback), this]
                          (cv::Mat&& img) mutable {
                c(std::move(img));
            };
            decode_hw(data, size, width, height,
                      make_copyable(std::move(decode)));
#else
            assert(false);
#endif
        } else {
            assert(false);
        }
    }

private:
    const Settings settings;
#ifdef USE_LIBVA
    struct HwContext;
    template<typename T>
    struct MoveHack {
        union {
            T val;
        };

        explicit MoveHack(T&& v):
            val(std::move(v)) {}
        MoveHack(const MoveHack&) { std::abort(); }
        MoveHack(MoveHack&& rhs):
            val(std::move(rhs.val)) {}
        ~MoveHack() {
            val.~T();
        }

        template<typename... Types>
        auto operator()(Types&&... args)
        ->decltype(val(std::forward<Types>(args)...)) {
            return val(std::forward<Types>(args)...);
        }
    };
    template<typename T>
    auto make_copyable(T&& val)
    ->MoveHack<typename std::remove_reference<T>::type> {
        return MoveHack<typename std::remove_reference<T>::type>{std::move(val)};
    }

    using callback_t = std::function<void(cv::Mat&&)>;

    std::unique_ptr<HwContext> hw_context;

    void decode_hw(const void* data, size_t size, unsigned width,
                   unsigned height, callback_t callback);
#endif
};
