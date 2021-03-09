// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <utility>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>

class VideoFrame {  // VideoFrame can represent not a single image but the whole grid
public:
    typedef std::shared_ptr<VideoFrame> Ptr;

    VideoFrame(unsigned sourceID, int64_t frameId, const cv::Mat& frame = cv::Mat()) :
        sourceID{sourceID}, frameId{frameId}, frame{frame} {}
    virtual ~VideoFrame() = default;  // A user has to define how it is reconstructed

    const unsigned sourceID;
    const int64_t frameId;
    cv::Mat frame;
};

class Worker;

class Task {
public:
    explicit Task(VideoFrame::Ptr sharedVideoFrame, float priority = 0):
        sharedVideoFrame{sharedVideoFrame}, priority{priority} {}
    virtual bool isReady() = 0;
    virtual void process() = 0;
    virtual ~Task() = default;

    VideoFrame::Ptr sharedVideoFrame;  // it is possible that two tasks try to draw on the same cvMat
    const float priority;
};

struct HigherPriority {
    bool operator()(const std::shared_ptr<Task>& lhs, const std::shared_ptr<Task>& rhs) const {
        return lhs->priority > rhs->priority
            || (lhs->priority == rhs->priority && lhs->sharedVideoFrame->frameId < rhs->sharedVideoFrame->frameId)
            || (lhs->priority == rhs->priority && lhs->sharedVideoFrame->frameId == rhs->sharedVideoFrame->frameId && lhs < rhs);
    }
};

class Worker {
public:
    explicit Worker(unsigned threadNum):
        threadPool(threadNum), running{false} {}
    ~Worker() {
        stop();
    }
    void runThreads() {
        running = true;
        for (std::thread& t : threadPool) {
            t = std::thread(&Worker::threadFunc, this);
        }
    }
    void push(std::shared_ptr<Task> task) {
        tasksMutex.lock();
        tasks.insert(task);
        tasksMutex.unlock();
        tasksCondVar.notify_one();
    }
    void threadFunc() {
        while (running) {
            std::unique_lock<std::mutex> lk(tasksMutex);
            while (running && tasks.empty()) {
                tasksCondVar.wait(lk);
            }
            try {
                auto it = std::find_if(tasks.begin(), tasks.end(), [](const std::shared_ptr<Task>& task){return task->isReady();});
                if (tasks.end() != it) {
                    const std::shared_ptr<Task> task = std::move(*it);
                    tasks.erase(it);
                    lk.unlock();
                    task->process();
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock{excpetionMutex};
                if (nullptr == currentException) {
                    currentException = std::current_exception();
                    stop();
                }
            }
        }
    }
    void stop() {
        running = false;
        tasksCondVar.notify_all();
    }
    void join() {
        for (auto& t : threadPool) {
            t.join();
        }
        if (nullptr != currentException) {
            std::rethrow_exception(currentException);
        }
    }

private:
    std::condition_variable tasksCondVar;
    std::set<std::shared_ptr<Task>, HigherPriority> tasks;
    std::mutex tasksMutex;
    std::vector<std::thread> threadPool;
    std::atomic<bool> running;
    std::exception_ptr currentException;
    std::mutex excpetionMutex;
};

void tryPush(const std::weak_ptr<Worker>& worker, std::shared_ptr<Task>&& task) {
    try {
        std::shared_ptr<Worker>(worker)->push(task);
    } catch (const std::bad_weak_ptr&) {}
}

template <class C> class ConcurrentContainer {
public:
    C container;
    mutable std::mutex mutex;

    bool lockedEmpty() const noexcept {
        std::lock_guard<std::mutex> lock{mutex};
        return container.empty();
    }
    typename C::size_type lockedSize() const noexcept {
        std::lock_guard<std::mutex> lock{mutex};
        return container.size();
    }
    void lockedPush_back(const typename C::value_type& value) {
        std::lock_guard<std::mutex> lock{mutex};
        container.push_back(value);
    }
    bool lockedTry_pop(typename C::value_type& value) {
        mutex.lock();
        if (!container.empty()) {
            value = container.back();
            container.pop_back();
            mutex.unlock();
            return true;
        } else {
            mutex.unlock();
            return false;
        }
    }

    operator C() const {
        std::lock_guard<std::mutex> lock{mutex};
        return container;
    }
};
