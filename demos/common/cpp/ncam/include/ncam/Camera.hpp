/*
 * Wahtari nApp Samples
 * Copyright (c) 2021 Wahtari GmbH
 *
 * All source code in this file is subject to the included LICENSE file.
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <VimbaCPP/Include/VimbaCPP.h>

#include "BufferedChannel.hpp"


namespace ncam {

// The Camera class is a thin wrapper around a vimba Camera.
// It encapsulates the setup and teardown code and provides easy to use methods
// to quickly get a camera up and running.
// This class is not thread-safe.
class Camera {
public:
    Camera();
    ~Camera();

    // printSystemVersion prints the semver version of the Vimba SDK.
    void printSystemVersion();

    // start opens the first found camera and starts the image acquisition on it.
    bool start(int numFrameBuffers);

    // stop stops the image acquisition of the started camera.
    // It is valid to call stop multiple times. 
    // The destructor makes sure to call stop as well.
    void stop();

    // read the camera frame from the queue.
    bool read(cv::Mat& dst, std::chrono::milliseconds timeout = std::chrono::milliseconds(std::chrono::seconds(3)));

private:
    AVT::VmbAPI::VimbaSystem& avtSystem_;
    AVT::VmbAPI::CameraPtr    avt_;

    std::shared_ptr<BufferedChannel<cv::Mat>> bufChan_;
    bool opened_;
    bool grabbing_;
};

} // End of namespace.
