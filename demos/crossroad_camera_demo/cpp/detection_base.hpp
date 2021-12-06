// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "openvino/openvino.hpp"
#include "utils/slog.hpp"

#pragma once

struct BaseDetection {
    ov::runtime::ExecutableNetwork net;
    ov::runtime::InferRequest request;
    std::string& commandLineFlag;
    std::string topoName;
    ov::runtime::Tensor input_tensor;
    std::string inputName;
    std::string outputName;

    BaseDetection(std::string& commandLineFlag, const std::string& topoName) :
        commandLineFlag(commandLineFlag), topoName(topoName) {}

    ov::runtime::ExecutableNetwork* operator->() {
        return &net;
    }

    virtual std::shared_ptr<ov::Function> read(const ov::runtime::Core& core) = 0;

    virtual void setRoiBlob(const ov::runtime::Tensor& roi_tensor) {
        if (!enabled())
            return;
        if (!request)
            request = net.create_infer_request();

        request.set_input_tensor(roi_tensor);
    }

    virtual void enqueue(const cv::Mat& person) {
        if (!enabled())
            return;
        if (!request)
            request = net.create_infer_request();

        input_tensor = request.get_input_tensor();
        matToTensor(person, input_tensor);
    }

    virtual void submitRequest() {
        if (!enabled() || !request)
            return;

        request.start_async();
    }

    virtual void wait() {
        if (!enabled()|| !request)
            return;

        request.wait();
    }

    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " detection DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) {}

    void into(ov::runtime::Core& core, const std::string& deviceName) const {
        if (detector.enabled()) {
            detector.net = core.compile_model(detector.read(core), deviceName);
            logExecNetworkInfo(detector.net, detector.commandLineFlag, deviceName, detector.topoName);
        }
    }
};
