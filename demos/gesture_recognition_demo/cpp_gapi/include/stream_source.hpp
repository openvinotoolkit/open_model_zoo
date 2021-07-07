// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utils/images_capture.h>
#include <opencv2/gapi.hpp>
#include <chrono>
#include <thread>
#include <mutex>

std::mutex batch_lock; // batch frames filling will be locked

namespace custom {
class BatchProducer {
public:
    BatchProducer(const int batch_size, const float batch_fps)
        : batch_fps(batch_fps) {
        /** Create batch memory space for batch_size + 2 size
         *  first additional element is fast image
         *  second additional Mat is sacriface of memory for data about first element
        **/
        batch = std::vector<cv::Mat>(batch_size + 1 + 1); // 16(8) 15FPS-batch imgaes + one fast image +  batch description
        batch[batch_size + 1].create(cv::Size{1, 1}, CV_8U);
    }
    std::vector<cv::Mat> getBatch() {
        return batch;
    }

    void fillFastFrame(const cv::Mat& frame) {
        /** Copy fast frame from VideoCapture to batch memory as 17th (9) image **/
        batch[batch.size() - 2] = frame.clone(); // 16th (from 0)
    }

    void fillBatch(const cv::Mat& frame, std::chrono::steady_clock::time_point time) {
        /** Place of new frame in batch **/
        const int step = updateStep(batch.size() - 2);
        batch_lock.lock();
        /** Adding of new image to batch. **/
        batch[step] = frame.clone();
        /** Putting of info about batch to additional element **/
        auto ptr = batch[batch.size() - 1].ptr<uint8_t>();
        ptr[0] = first_el;
        batch_lock.unlock();
        const auto cur_step = std::chrono::steady_clock::now() - time;
        const auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(cur_step).count();
        int time_step = int(1000.f / batch_fps);
        if (gap < time_step) {
            std::this_thread::sleep_for(std::chrono::milliseconds(time_step - gap));
        }
    }
private:
    float batch_fps = 0; // constant FPS for batc
    std::vector<cv::Mat> batch; // pack of images for graph
    size_t first_el = 0; // place of first image of batch
    size_t images_in_batch_count = 0; // amount of images in batch

    int updateStep(const size_t batch_size) {
        if (images_in_batch_count < batch_size) {
            /** case when batch isn't filled **/
            return images_in_batch_count++;
        } else {
            /** Cyclic buffer if filled. Counting of step for next image **/
            if (first_el > (batch_size - 1)) {
                first_el = 0; // case when new image has batch_size - 1 place (last in batch memory)
            }
            return first_el++;
        }
    }
};

void runBatchFill(const cv::Mat& frame,
                        BatchProducer& producer,
                        std::chrono::steady_clock::time_point& time) {
    while(!frame.empty()) {
        producer.fillBatch(frame, time);
    }
}

class CustomCapSource : public cv::gapi::wip::IStreamSource
{
public:
    explicit CustomCapSource(const std::shared_ptr<ImagesCapture>& cap,
                             const cv::Size& frame_size,
                             const int batch_size,
                             const float batch_fps)
        : cap(cap), producer(batch_size, batch_fps) {
        /** Create and get first image for batch **/
        GAPI_Assert(first_batch.empty());
        if (batch_size == 0 || batch_size == 1) {
            GAPI_Assert(false && "Batch must contain more than one image");
        }

        fast_frame.create(frame_size, CV_8UC3);

        /** Reading of frame with ImagesCapture class **/
        read_time = std::chrono::steady_clock::now();
        fast_frame = cap->read();
        if (!fast_frame.data) {
            GAPI_Assert(false && "Couldn't grab the frame");
        }

        /** Batch filling with constant time step **/
        fill_bath_thr.detach();

        producer.fillFastFrame(fast_frame);
        first_batch = producer.getBatch();
    }

protected:
    std::shared_ptr<ImagesCapture> cap; // wrapper for cv::VideoCapture
    BatchProducer producer; // class batch-construcor
    bool first_pulled = false; // is first already pulled
    std::vector<cv::Mat> first_batch; // batch from constructor
    cv::Mat fast_frame; // frame from cv::VideoCapture
    std::chrono::steady_clock::time_point read_time; // timepoint from cv::read()
    std::thread fill_bath_thr = std::thread(runBatchFill,
                                            std::ref(fast_frame),
                                            std::ref(producer),
                                            std::ref(read_time));

    virtual bool pull(cv::gapi::wip::Data &data) override {
        /** Is first already pulled **/
        if (!first_pulled) {
            GAPI_Assert(!first_batch.empty());
            first_pulled = true;
            cv::detail::VectorRef ref(std::move(first_batch));
            data = std::move(ref);
            return true;
        }

        /** Frame reading with ImagesCapture class **/
        read_time = std::chrono::steady_clock::now();
        fast_frame = cap->read();
        if (!fast_frame.data) {
            return false;
        }

        /** Put fast frame to the batch **/
        producer.fillFastFrame(fast_frame);

        /** Put pulled batch to GRunArg data **/
        cv::detail::VectorRef ref(producer.getBatch());
        data = std::move(ref);
        return true;
    }

    virtual cv::GMetaArg descr_of() const override {
        GAPI_Assert(!first_batch.empty());
        return cv::GMetaArg{ cv::empty_array_desc() };
    }
};

} // namespace custom
