// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "openvino/openvino.hpp"
#include "utils/slog.hpp"

#pragma once

struct BaseDetection {
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_infer_request;
    std::string& m_commandLineFlag;
    std::string m_detectorName;
    ov::Tensor m_input_tensor;
    std::string m_inputName;
    std::string m_outputName;

    BaseDetection(std::string& commandLineFlag, const std::string& detectorName) :
        m_commandLineFlag(commandLineFlag), m_detectorName(detectorName) {}

    ov::CompiledModel* operator->() {
        return &m_compiled_model;
    }

    virtual std::shared_ptr<ov::Model> read(const ov::Core& core) = 0;

    virtual void setRoiTensor(const ov::Tensor& roi_tensor) {
        if (!enabled())
            return;
        if (!m_infer_request)
            m_infer_request = m_compiled_model.create_infer_request();

        m_infer_request.set_input_tensor(roi_tensor);
    }

    virtual void enqueue(const cv::Mat& person) {
        if (!enabled())
            return;
        if (!m_infer_request)
            m_infer_request = m_compiled_model.create_infer_request();

        m_input_tensor = m_infer_request.get_input_tensor();
        matToTensor(person, m_input_tensor);
    }

    virtual void submitRequest() {
        if (!enabled() || !m_infer_request)
            return;

        m_infer_request.start_async();
    }

    virtual void wait() {
        if (!enabled()|| !m_infer_request)
            return;

        m_infer_request.wait();
    }

    mutable bool m_enablingChecked = false;
    mutable bool m_enabled = false;

    bool enabled() const  {
        if (!m_enablingChecked) {
            m_enabled = !m_commandLineFlag.empty();
            if (!m_enabled) {
                slog::info << m_detectorName << " detection DISABLED" << slog::endl;
            }
            m_enablingChecked = true;
        }
        return m_enabled;
    }
};

struct Load {
    BaseDetection& m_detector;
    explicit Load(BaseDetection& detector) : m_detector(detector) {}

    void into(ov::Core& core, const std::string& deviceName) const {
        if (m_detector.enabled()) {
            m_detector.m_compiled_model = core.compile_model(m_detector.read(core), deviceName);
            logCompiledModelInfo(m_detector.m_compiled_model, m_detector.m_commandLineFlag, deviceName, m_detector.m_detectorName);
        }
    }
};
