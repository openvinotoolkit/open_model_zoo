// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <set>
#include <thread>
#include <vector>
#include <queue>

#include <opencv2/opencv.hpp>

class InputChannel;

class IInputSource {
public:
    virtual bool read(cv::Mat& mat, const std::shared_ptr<InputChannel>& caller) = 0;
    virtual void addSubscriber(const std::weak_ptr<InputChannel>& inputChannel) = 0;
    virtual cv::Size getSize() = 0;
    virtual void lock() {
        sourceLock.lock();
    }
    virtual void unlock() {
        sourceLock.unlock();
    }
    virtual ~IInputSource() = default;
private:
    std::mutex sourceLock;
};

class InputChannel: public std::enable_shared_from_this<InputChannel> {  // note: public inheritance
public:
    InputChannel(const InputChannel&) = delete;
    InputChannel& operator=(const InputChannel&) = delete;
    static std::shared_ptr<InputChannel> create(const std::shared_ptr<IInputSource>& source) {
        auto tmp = std::shared_ptr<InputChannel>(new InputChannel(source));
        source->addSubscriber(tmp);
        return tmp;
    }
    bool read(cv::Mat& mat) {
        readQueueMutex.lock();
        if (readQueue.empty()) {
            readQueueMutex.unlock();
            source->lock();
            readQueueMutex.lock();
            if (readQueue.empty()) {
                bool res = source->read(mat, shared_from_this());
                readQueueMutex.unlock();
                source->unlock();
                return res;
            } else {
                source->unlock();
            }
        }
        mat = readQueue.front().clone();
        readQueue.pop();
        readQueueMutex.unlock();
        return true;
    }
    void push(const cv::Mat& mat) {
        readQueueMutex.lock();
        readQueue.push(mat);
        readQueueMutex.unlock();
    }
    cv::Size getSize() {
        return source->getSize();
    }

private:
    explicit InputChannel(const std::shared_ptr<IInputSource>& source): source{source} {}
    std::shared_ptr<IInputSource> source;
    std::queue<cv::Mat, std::list<cv::Mat>> readQueue;
    std::mutex readQueueMutex;
};

class VideoCaptureSource: public IInputSource {
public:
    VideoCaptureSource(const cv::VideoCapture& videoCapture, bool loop): videoCapture{videoCapture}, loop{loop},
        imSize{static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT))} {}
    bool read(cv::Mat& mat, const std::shared_ptr<InputChannel>& caller) override {
        if (!videoCapture.read(mat)) {
            if (loop) {
                videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
                videoCapture.read(mat);
            } else {
                return false;
            }
        }
        if (1 != subscribedInputChannels.size()) {
            cv::Mat shared = mat.clone();
            for (const std::weak_ptr<InputChannel>& weakInputChannel : subscribedInputChannels) {
                try {
                    std::shared_ptr<InputChannel> sharedInputChannel = std::shared_ptr<InputChannel>(weakInputChannel);
                    if (caller != sharedInputChannel) {
                        sharedInputChannel->push(shared);
                    }
                } catch (const std::bad_weak_ptr&) {}
            }
        }
        return true;
    }
    void addSubscriber(const std::weak_ptr<InputChannel>& inputChannel) override {
        subscribedInputChannels.push_back(inputChannel);
    }
    cv::Size getSize() override {
        return imSize;
    }

private:
    std::vector<std::weak_ptr<InputChannel>> subscribedInputChannels;
    cv::VideoCapture videoCapture;
    bool loop;
    cv::Size imSize;
};

class ImageSource: public IInputSource {
public:
    ImageSource(const cv::Mat& im, bool loop): im{im.clone()}, loop{loop} {}  // clone to avoid image changing
    bool read(cv::Mat& mat, const std::shared_ptr<InputChannel>& caller) override {
        if (!loop) {
            auto subscribedInputChannelsIt = subscribedInputChannels.find(caller);
            if (subscribedInputChannels.end() == subscribedInputChannelsIt) {
                return false;
            } else {
                subscribedInputChannels.erase(subscribedInputChannelsIt);
                mat = im;
                return true;
            }
        } else {
            mat = im;
            return true;
        }
    }
    void addSubscriber(const std::weak_ptr<InputChannel>& inputChannel) override {
        if (false == subscribedInputChannels.insert(inputChannel).second)
            throw std::invalid_argument("The insertion did not take place");
    }
    cv::Size getSize() override {
        return im.size();
    }

private:
    std::set<std::weak_ptr<InputChannel>, std::owner_less<std::weak_ptr<InputChannel>>> subscribedInputChannels;
    cv::Mat im;
    bool loop;
};
