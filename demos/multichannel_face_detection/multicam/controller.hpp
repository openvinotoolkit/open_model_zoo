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
