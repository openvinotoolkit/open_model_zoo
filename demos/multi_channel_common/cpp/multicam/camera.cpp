// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "camera.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>
#include <memory>
#include <utility>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "controller.hpp"
#include "utils.hpp"

namespace mcam {
namespace {
int xioctl(int fd, unsigned long int request, void *arg) {
    int r = -1;
    do {
        r = ioctl(fd, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

file_descriptor open_device(string_ref name) {
    assert(nullptr != name);
    file_descriptor fd(open(name.data(), O_RDWR | O_NONBLOCK, 0));

    if (!fd.valid()) {
        std::string str = std::string("cannot open camera device: \"")
                          + name.data() + "\"";
        throw_error(str);
    }

    struct stat st = {};
    if (-1 == fstat(fd.get(), &st)) {
        throw_errno_error("cannot identify camera device:", errno);
    }

    if (!S_ISCHR(st.st_mode)) {
        throw_errno_error("file is not a device:", errno);
    }
    return fd;
}

void set_device_params(const file_descriptor& fd, camera::settings& params,
                       std::size_t& frame_buffer_size) {
    v4l2_capability cap = {};
    if (-1 == xioctl(fd.get(), VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            throw_error("not a V4L2 device");
        } else {
            throw_errno_error("cannot query device capabilities:", errno);
        }
    }

    if (0 == (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        throw_error("not a capture device");
    }
    if (0 == (cap.capabilities & V4L2_CAP_STREAMING)) {
        throw_error("device doesn't support streaming");
    }

//    v4l2_cropcap cropcap = {};

    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = params.width;
    fmt.fmt.pix.height      = params.height;
    fmt.fmt.pix.pixelformat = params.format4cc;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;

    if (-1 == xioctl(fd.get(), VIDIOC_S_FMT, &fmt)) {
        throw_errno_error("Unable to set format:", errno);
    }
    params.width     = fmt.fmt.pix.width;
    params.height    = fmt.fmt.pix.height;
    params.format4cc = fmt.fmt.pix.pixelformat;

    if (0 != params.frametime_numerator ||
        0 != params.frametime_denominator) {
        v4l2_streamparm parm = {};

        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        parm.parm.capture.timeperframe.numerator
                = params.frametime_numerator;
        parm.parm.capture.timeperframe.denominator
                = params.frametime_denominator;
        if (-1 == xioctl(fd.get(), VIDIOC_S_PARM, &parm)) {
            throw_errno_error("Unable to set framerate:", errno);
        }
        params.frametime_numerator
                = parm.parm.capture.timeperframe.numerator;
        params.frametime_denominator
                = parm.parm.capture.timeperframe.denominator;
    }
    frame_buffer_size = static_cast<std::size_t>(fmt.fmt.pix.sizeimage);
}
}  // namespace

camera::camera(controller& owner_, string_ref name_, callback_t callback_,
               const settings& params_):
    owner(owner_),
    params(params_),
    dev(open_device(name_)),
    callback(std::move(callback_)) {
    assert(nullptr != callback);
    set_device_params(dev, params, frame_buffer_size);
    alloc_buffers();
    start_capture();
    owner.register_camera(*this);
}

camera::~camera() {
    owner.unregister_camera(*this);
}

void camera::alloc_buffers() {
    assert(frame_buffer_size > 0);
    assert(params.num_buffers > 0);
    assert(dev.valid());
    v4l2_requestbuffers req = {};

    req.count  = params.num_buffers;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (-1 == xioctl(dev.get(), VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            throw_error("User pointer i/o not supported");
        } else {
            throw_errno_error("Unable to setup user pointer i/o mode:", errno);
        }
    }
    params.num_buffers = req.count;

    buffers.clear();
    const auto count = params.num_buffers;
    buffers.reserve(count);
    for (unsigned i = 0 ; i < count; ++i) {
        std::unique_ptr<char[]> buff(new char[frame_buffer_size]);
        buffers.emplace_back(std::move(buff));
    }

    for (unsigned i = 0 ; i < count; ++i) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        buf.index = i;
        buf.m.userptr = reinterpret_cast<unsigned long>(buffers[i].get());
        buf.length = static_cast<__u32>(frame_buffer_size);

        if (-1 == xioctl(dev.get(), VIDIOC_QBUF, &buf)) {
            throw_errno_error("Unable to enqueue frame buffer ptr:", errno);
        }
    }
}

void camera::start_capture() {
    assert(dev.valid());
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(dev.get(), VIDIOC_STREAMON, &type)) {
        throw_errno_error("Unable to start capture:", errno);
    }
}

void camera::read_frame() {
    assert(dev.valid());
    assert(nullptr != callback);
    while (true) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        if (-1 == xioctl(dev.get(), VIDIOC_DQBUF, &buf)) {
            switch (errno) {
            case EAGAIN:
                return;

            case EIO:
            case ENODEV:
                callback(frame_status::failure, params, frame{});
                return;

            default:
                throw_errno_error("Unable to get frame buffer ptr:", errno);
            }
        }
        auto ptr = reinterpret_cast<void*>(buf.m.userptr);
        assert(nullptr != ptr);
        auto len = buf.bytesused;
        assert(len > 0);
        callback(frame_status::ok, params, frame(*this, buf.index, ptr, len));
    }
}

void camera::reclaim_frame(frame& f) {
    v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_USERPTR;
    buf.index = f.index;
    buf.m.userptr = reinterpret_cast<unsigned long>(f.ptr);
    buf.length = static_cast<__u32>(frame_buffer_size);

    if (-1 == xioctl(dev.get(), VIDIOC_QBUF, &buf)) {
        throw_errno_error("Unable to enqueue frame buffer ptr:", errno);
    }
}

camera::frame::frame(camera& c, unsigned i, void* p, std::size_t l):
    cam(&c), index(i), ptr(p), len(l) {
    assert(nullptr != ptr);
    assert(0 != len);
}

camera::frame::frame(camera::frame&& rhs) {
    std::swap(cam, rhs.cam);
    std::swap(index, rhs.index);
    std::swap(ptr, rhs.ptr);
    std::swap(len, rhs.len);
}

camera::frame::~frame() {
    if (valid()) {
        cam->reclaim_frame(*this);
    }
}

camera::frame& camera::frame::operator=(camera::frame&& rhs) {
    if (this != &rhs) {
        std::swap(cam, rhs.cam);
        std::swap(index, rhs.index);
        std::swap(ptr, rhs.ptr);
        std::swap(len, rhs.len);
    }
    return *this;
}

bool camera::frame::valid() const {
    return nullptr != cam;
}

const void* camera::frame::data() const {
    return ptr;
}

std::size_t camera::frame::size() const {
    return len;
}

}  // namespace mcam
