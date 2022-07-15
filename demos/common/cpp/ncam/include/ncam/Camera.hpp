/*
 * Wahtari nApp Samples
 * Copyright (c) 2021 Wahtari GmbH
 *
 * All source code in this file is subject to the included LICENSE file.
 */

#pragma once

#include <atomic>
#include <functional>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <VimbaCPP/Include/VimbaCPP.h>

namespace ncam {

typedef std::function<void (const cv::Mat&)> FrameCallback;

// The FrameObserver class implements the vimba IFrameObserver interface and provides
// a callback to handle new frames read off of the camera.
class FrameObserver : public IFrameObserver {
public:
    FrameObserver(CameraPtr cam, VmbPixelFormatType pxFmt, FrameCallback cb);

    // FrameReceived is the callback that handles newly read frames.
    // We convert the frame to an OpenCV Mat and distribute it to both the 
    // jpeg encoding routines, as well as the inference routines.
    void FrameReceived(const FramePtr frame);

private:
    AVT::VmbAPICameraPtr          cam_;
    AVT::VmbAPIVmbPixelFormatType pxFmt_;

    FrameCallback cb_;
};

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
    bool start(int numFrameBuffers, FrameCallback cb);

    // stop stops the image acquisition of the started camera.
    // It is valid to call stop multiple times. 
    // The destructor makes sure to call stop as well.
    void stop();

private:
    AVT::VmbAPIVimbaSystem& avtSystem_;
    AVT::VmbAPICameraPtr    avt_;

    bool opened_;
    bool grabbing_;
};

} // End of namespace.
