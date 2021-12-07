// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>
#include <string>

#include "openvino/openvino.hpp"

#include "utils/slog.hpp"
#include "detection_base.hpp"
#include "crossroad_camera_demo.hpp"

using namespace ov::preprocess;

struct PersonDetection : BaseDetection {
    size_t maxProposalCount;
    size_t objectSize;
    float width = 0.0f;
    float height = 0.0f;
    bool resultsFetched = false;

    struct Result {
        int label = 0;
        float confidence = .0f;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void setRoiBlob(const ov::runtime::Tensor& roi_tensor) override {
        ov::Shape shape = roi_tensor.get_shape();
        ov::Layout layout("NHWC");

        height = static_cast<float>(roi_tensor.get_shape()[ov::layout::height_idx(layout)]);
        width = static_cast<float>(roi_tensor.get_shape()[ov::layout::width_idx(layout)]);
        BaseDetection::setRoiBlob(roi_tensor);
    }

    void enqueue(const cv::Mat& frame) override {
        height = static_cast<float>(frame.rows);
        width = static_cast<float>(frame.cols);
        BaseDetection::enqueue(frame);
    }

    PersonDetection() : BaseDetection(FLAGS_m, "Person Detection"), maxProposalCount(0), objectSize(0) {}

    std::shared_ptr<ov::Function> read(const ov::runtime::Core& core) override {
        // Read network model
        auto network = core.read_model(FLAGS_m);
        // Set batch size to 1
        const ov::Layout layout_nhwc{"NHWC"};
        ov::Shape input_shape = network->input().get_shape();
        input_shape[ov::layout::batch_idx(layout_nhwc)] = 1;
        network->reshape({{network->input().get_any_name(), input_shape}});

        // SSD-based network should have one input and one output
        // Check inputs
        ov::OutputVector inputs = network->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Person Detection network should have only one input");
        }

        inputName = network->input().get_any_name();

        // Check outputs
        ov::OutputVector outputs = network->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("Person Detection network should have only one output");
        }

        outputName = network->output().get_any_name();

        maxProposalCount = network->output().get_shape()[2];
        objectSize = network->output().get_shape()[3];

        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputs[0].get_shape().size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }

        const ov::Layout tensor_layout{ "NHWC" };

        ov::preprocess::PrePostProcessor ppp = PrePostProcessor(network);

        if (FLAGS_auto_resize) {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_spatial_dynamic_shape().
                set_layout(tensor_layout);
            ppp.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW").
                resize(ResizeAlgorithm::RESIZE_LINEAR);
            ppp.input().model().set_layout("NCHW");
            ppp.output().tensor().set_element_type(ov::element::f32);
        } else {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }

        network = ppp.build();

        return network;
    }

    void fetchResults() {
        if (!enabled())
            return;

        results.clear();

        if (resultsFetched)
            return;

        resultsFetched = true;
        const float* detections = request.get_output_tensor().data<float>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0]; // in case of batch
            if (image_id < 0) {
                // end of detections
                break;
            }

            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];

            r.location.x = static_cast<int>(detections[i * objectSize + 3] * width);
            r.location.y = static_cast<int>(detections[i * objectSize + 4] * height);
            r.location.width = static_cast<int>(detections[i * objectSize + 5] * width - r.location.x);
            r.location.height = static_cast<int>(detections[i * objectSize + 6] * height - r.location.y);

            if (r.confidence <= FLAGS_t) {
                continue;
            }

            if (FLAGS_r) {
                slog::debug <<
                    "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                    "    (" << r.location.x << "," << r.location.y <<
                    ")-(" << r.location.width << "," << r.location.height << ")" <<
                    ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << slog::endl;
            }
            results.push_back(r);
        }
    }
};
