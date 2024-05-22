// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/gapi.hpp>

#include <utils/images_capture.h>
#include <utils/slog.hpp>

namespace custom {
class BatchProducer {
public:
    BatchProducer(const int batch_size, const float batch_fps, const std::shared_ptr<bool>& drop_batch)
        : batch_fps(batch_fps),
          drop_batch(drop_batch) {
        /** Create batch memory space for batch_size + 2 size
         *  first additional element is fast image
         *  second additional Mat is sacriface of memory for data about first element
         **/
        batch =
            std::vector<cv::Mat>(batch_size + 1 + 1);  // 16(8) 15FPS-batch imgaes + one fast image +  batch description
        batch[batch_size + 1].create(cv::Size{1, 2},
                                     CV_8U);  // 1x2 Mat for first element position and is_filled batch state
        auto ptr = batch[batch.size() - 1].ptr<uint8_t>();
        ptr[1] = 0;  // set is_filled to NO
    }
    std::vector<cv::Mat> getBatch() {
        batch_lock.lock();
        std::vector<cv::Mat> temp_batch = batch;
        batch_lock.unlock();
        return temp_batch;
    }

    void fillFastFrame(const cv::Mat& frame) {
        /** Copy fast frame from VideoCapture to batch memory as 17th (9) image **/
        frame.copyTo(batch[batch.size() - 2]);  // 16th (from 0)
    }

    void fillBatch(const cv::Mat& frame, std::chrono::steady_clock::time_point time) {
        if (*drop_batch) {
            DropBatchInfo();
        }
        /** Place of new frame in batch **/
        const int step = updateStep(batch.size() - 2);
        /** Adding of new image to batch. **/
        frame.copyTo(batch[step]);
        /** Putting of info about batch to additional element **/
        auto ptr = batch[batch.size() - 1].ptr<uint8_t>();
        ptr[0] = first_el;  // position of start of batch in cyclic buffer
        const auto cur_step = std::chrono::steady_clock::now() - time;
        const auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(cur_step);
        const auto time_step = std::chrono::milliseconds(static_cast<int>(1000.f / batch_fps));  // 1/15 sec
        if (gap < time_step) {
            std::this_thread::sleep_for(time_step - gap);  // wait for constant step of batch update
        }
    }

private:
    float batch_fps = 0;  // constant FPS for batch
    const std::shared_ptr<bool> drop_batch;  // drop parameters of batch
    std::vector<cv::Mat> batch;  // pack of images for graph
    size_t first_el = 0;  // place of first image in batch
    size_t images_in_batch_count = 0;  // number of images in batch
    bool is_filled = false;  // is batch filled
    std::mutex batch_lock;  // batch frames filling will be locked

    int updateStep(const size_t batch_size) {
        if (images_in_batch_count < batch_size) {
            /** case when batch isn't filled **/
            return images_in_batch_count++;
        } else {
            if (!is_filled) {
                auto ptr = batch[batch.size() - 1].ptr<uint8_t>();
                ptr[1] = 1;
                is_filled = true;
            }
            /** Cyclic buffer if filled. Counting of step for next image **/
            first_el = (first_el + 1) % batch_size;
            return first_el;
        }
    }

    void DropBatchInfo() {
        /** Drop batch information.
         * Processing will continue when the batch will be filled
         * Data of the batch will be overwritten */
        auto ptr = batch[batch.size() - 1].ptr<uint8_t>();
        ptr[0] = 0;  // first position
        ptr[1] = 0;  // batch if filled
        first_el = 0;
        images_in_batch_count = 0;
        is_filled = false;
    }
};

static void runBatchFill(const cv::Mat& frame,
                         BatchProducer& producer,
                         std::chrono::steady_clock::time_point& time,
                         bool& is_filling_possible) {
    while (is_filling_possible) {
        producer.fillBatch(frame, time);
    }
}

class GestRecCapSource : public cv::gapi::wip::IStreamSource {
public:
    explicit GestRecCapSource(const std::shared_ptr<ImagesCapture>& cap,
                              const cv::Size& frame_size,
                              const int batch_size,
                              const float batch_fps,
                              const std::shared_ptr<bool>& drop_batch)
        : cap(cap),
          producer(batch_size, batch_fps, drop_batch),
          source_fps(cap->fps()) {
        if (source_fps <= 0.) {
            source_fps = 30.;
            wait_gap = true;
            slog::warn << "Got a non-positive value as FPS of the input. Interpret it as 30 FPS" << slog::endl;
        }
        /** Create and get first image for batch **/
        if (!first_batch.empty()) {
            throw std::runtime_error("first_batch must be empty before creation");
        }
        if (batch_size == 0 || batch_size == 1) {
            throw std::runtime_error("Batch must contain more than one image");
        }

        /** Reading of frame with ImagesCapture class **/
        read_time = std::chrono::steady_clock::now();
        cv::Mat fast_frame = cap->read();
        if (!fast_frame.data) {
            throw std::runtime_error("Couldn't grab the frame");
        }
        producer.fillFastFrame(fast_frame);
        fast_frame.copyTo(thread_frame);
        /** Batch filling with constant time step **/
        std::thread fill_bath_thr(runBatchFill,
                                  std::cref(thread_frame),
                                  std::ref(producer),
                                  std::ref(read_time),
                                  std::ref(is_filling_possible));
        fill_bath_thr.detach();
        first_batch = producer.getBatch();
    }

protected:
    std::shared_ptr<ImagesCapture> cap;  // wrapper for cv::VideoCapture
    BatchProducer producer;  // class batch-construcor
    double source_fps = 0.;  // input source framerate
    bool wait_gap = false;  // waiting for fast frame reading (stop main thread when got a non-positive FPS value)
    bool first_pulled = false;  // is first already pulled
    std::vector<cv::Mat> first_batch;  // batch from constructor
    cv::Mat thread_frame;  // frame for batch constant filling
    std::mutex thread_frame_lock;  // lock for shared frame
    std::chrono::steady_clock::time_point read_time;  // timepoint from cv::read()
    bool is_filling_possible = true;  // access for batch filling

    bool pull(cv::gapi::wip::Data& data) override {
        /** Is first already pulled **/
        if (!first_pulled) {
            if (first_batch.empty()) {
                throw std::runtime_error("GestRecCapSource::pull() have got empty first_batch");
            }
            first_pulled = true;
            cv::detail::VectorRef ref(std::move(first_batch));
            data = std::move(ref);
            return true;
        }

        /** Frame reading with ImagesCapture class **/
        read_time = std::chrono::steady_clock::now();
        cv::Mat fast_frame = cap->read();

        if (!fast_frame.data) {
            is_filling_possible = false;
            return false;
        }

        thread_frame_lock.lock();
        fast_frame.copyTo(thread_frame);
        thread_frame_lock.unlock();

        /** Put fast frame to the batch **/
        producer.fillFastFrame(fast_frame);
        if (wait_gap) {
            const auto cur_step = std::chrono::steady_clock::now() - read_time;
            const auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(cur_step);
            const auto time_step = std::chrono::milliseconds(static_cast<int>(1000.f / source_fps));
            if (gap < time_step) {
                std::this_thread::sleep_for(time_step - gap);
            }
        }

        /** Put pulled batch to GRunArg data **/
        cv::detail::VectorRef ref(producer.getBatch());
        data = std::move(ref);
        return true;
    }

    cv::GMetaArg descr_of() const override {
        if (first_batch.empty()) {
            throw std::runtime_error("GestRecCapSource::descr_of() have got empty first_batch");
        }
        return cv::GMetaArg{cv::empty_array_desc()};
    }
};
}  // namespace custom
