// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "controller.hpp"

#include <vector>

#include "utils.hpp"

#include <fcntl.h>
#include <poll.h>

namespace mcam {
namespace {
using lock_guard = std::lock_guard<std::mutex>;
}

controller::controller() {
    queue_thread = std::thread([this]() {
        std::vector<pollfd> fds;
        std::vector<camera*> temp_ptrs;
        const auto poll_flags = POLLIN | POLLRDNORM | POLLERR;
        while (!terminate) { {
                lock_guard lock(list_mutex);
                auto size = cameras.size() + 1;
                if (fds.size() != size) {
                    fds.clear();
                    temp_ptrs.clear();
                    fds.push_back(pollfd{notifier.get_fd_to_poll(), POLLIN, 0});
                    for (auto& cam : cameras) {
                        assert(cam.dev.valid());
                        fds.push_back(pollfd{cam.dev.get(), poll_flags, 0});
                        temp_ptrs.push_back(&cam);
                    }
                }
            }

            if (-1 == poll(fds.data(), fds.size(), -1)) {
                throw_errno_error("failed wait on poll:", errno);
            }

            for (size_t i = 0; i < fds.size(); ++i) {
                if (0 != (fds[i].revents & poll_flags)) {
                    if (0 == i) {
                        notifier.flush();
                    } else {
                        auto cam = temp_ptrs[i - 1];
                        assert(nullptr != cam);
                        cam->read_frame();
                    }
                }
            }
        }
    });
}

controller::~controller() {
    terminate = true;
    notifier.signal();
    if (queue_thread.joinable()) {
        queue_thread.join();
    }
}

void controller::register_camera(camera& cam) {
    {
        lock_guard lock(list_mutex);
        cameras.push_back(cam);
    }
    notifier.signal();
}

void controller::unregister_camera(camera& cam) {
    {
        lock_guard lock(list_mutex);
        cameras.erase(decltype(cameras)::s_iterator_to(cam));
    }
    notifier.signal();
}

controller::poll_notifier::poll_notifier() {
    int pfd[2] = {};
    if (-1 == pipe(pfd)) {
        throw_errno_error("failed to create a pipe:", errno);
    }
    read_fd  = file_descriptor(pfd[0]);
    write_fd = file_descriptor(pfd[1]);
    int flags = fcntl(read_fd.get(), F_GETFL, 0);
    if (-1 == fcntl(read_fd.get(), F_SETFL, flags | O_NONBLOCK)) {
        throw_errno_error("Unable to set pipe ijnto non-blocking mode:", errno);
    }
}

void controller::poll_notifier::signal() {
    assert(write_fd.valid());
    char buff[1] = {42};
    if (-1 == write(write_fd.get(), buff, 1)) {
        throw_errno_error("failed to write to pipe:", errno);
    }
}

void controller::poll_notifier::flush() {
    assert(read_fd.valid());
    constexpr const size_t Len = 1024;
    char buff[Len];
    while (true) {
        auto res = read(read_fd.get(), buff, Len);
        if (-1 == res) {
            if (EAGAIN == errno) {
                break;
            }
            throw_errno_error("failed to read from pipe:", errno);
        }
        if (0 == res) {
            break;
        }
    }
}

}  // namespace mcam
