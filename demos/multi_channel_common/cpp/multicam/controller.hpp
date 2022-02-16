// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <mutex>
#include <thread>

#include "camera.hpp"
#include "utils.hpp"

namespace mcam {

class controller final {
public:
    friend class ::mcam::camera;

    controller();
    ~controller();

private:
    void register_camera(camera& cam);
    void unregister_camera(camera& cam);

    std::thread queue_thread;
    std::atomic_bool terminate = {false};

    std::mutex list_mutex;
    boost::intrusive::list<
        camera,
        boost::intrusive::member_hook<
            camera,
            boost::intrusive::list_member_hook<>, &camera::list_node
        >
    > cameras;

    struct poll_notifier final {
        file_descriptor read_fd;
        file_descriptor write_fd;

        poll_notifier();

        void signal();
        void flush();

        int get_fd_to_poll() const { return read_fd.get(); }
    };

    poll_notifier notifier;
};

}  // namespace mcam
