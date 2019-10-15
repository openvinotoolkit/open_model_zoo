// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

#include <boost/intrusive/list.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace mcam {

class controller;

class camera final {
public:
    friend class ::mcam::controller;
    struct settings final {
        unsigned width = 0;
        unsigned height = 0;
        unsigned format4cc = 0;

        /// Requested frame time (e.g. 1 / 60 for 60 fps)
        unsigned frametime_numerator = 0;
        unsigned frametime_denominator = 0;

        unsigned num_buffers = 1;
    };

    class frame final {
        camera* cam = nullptr;
        unsigned index = 0;
        void* ptr = nullptr;
        std::size_t len = 0;

        frame(camera& c, unsigned i, void* p, std::size_t l);
    public:
        friend class camera;

        frame() = default;
        frame(const frame&) = delete;
        frame(frame&& rhs);
        ~frame();

        frame& operator=(const frame&) = delete;
        frame& operator=(frame&& rhs);

        bool valid() const;

        const void* data() const;
        std::size_t size() const;
    };
    enum class frame_status {
        ok,
        failure
    };
    using callback_t
        = std::function<void(frame_status, const settings&, frame)>;

    camera(controller& owner_, string_ref name_, callback_t callback_,
           const settings& params_);
    ~camera();

private:
    friend class camera::frame;
    struct device {
        explicit device(const char* name);
        device(const device&) = delete;
        ~device();
        int fd = -1;
    };

    void alloc_buffers();
    void start_capture();
    void read_frame();
    void reclaim_frame(frame& f);

    controller& owner;
    settings params;
    file_descriptor dev;
    std::size_t frame_buffer_size = 0;
    std::vector<std::unique_ptr<char[]>> buffers;
    callback_t callback;

    boost::intrusive::list_member_hook<> list_node;
};

}  // namespace mcam
