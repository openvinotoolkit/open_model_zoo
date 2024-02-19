// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/openvino.hpp"

#include "gflags/gflags.h"
#include "utils/slog.hpp"
#include "detection_base.hpp"

struct PersonReIdentification : BaseDetection {
    std::vector<std::vector<float>> globalReIdVec; // contains vectors characterising all detected persons

    PersonReIdentification() : BaseDetection(FLAGS_m_reid, "Person Re-Identification Retail") {}

    unsigned long int findMatchingPerson(const std::vector<float>& newReIdVec) {
        auto size = globalReIdVec.size();

        // assigned REID is index of the matched vector from the globalReIdVec
        for (size_t i = 0; i < size; ++i) {
            float cosSim = cosineSimilarity(newReIdVec, globalReIdVec[i]);
            if (FLAGS_r) {
                slog::debug << "cosineSimilarity: " << cosSim << slog::endl;
            }
            if (cosSim > FLAGS_t_reid) {
                // We substitute previous person's vector by a new one characterising
                // last person's position
                globalReIdVec[i] = newReIdVec;
                return i;
            }
        }
        globalReIdVec.push_back(newReIdVec);
        return size;
    }

    std::vector<float> getReidVec() {
        ov::Tensor attribsTensor = m_infer_request.get_tensor(m_outputName);

        auto numOfChannels = attribsTensor.get_shape()[1];
        auto outputValues = attribsTensor.data<float>();

        return std::vector<float>(outputValues, outputValues + numOfChannels);
    }

    template <typename T>
    float cosineSimilarity(const std::vector<T>& vecA, const std::vector<T>& vecB) {
        if (vecA.size() != vecB.size()) {
            throw std::logic_error("cosine similarity can't be called for the vectors of different lengths: "
                                   "vecA size = " + std::to_string(vecA.size()) +
                                   "vecB size = " + std::to_string(vecB.size()));
        }

        T mul, denomA, denomB, A, B;
        mul = denomA = denomB = A = B = 0;
        for (size_t i = 0; i < vecA.size(); ++i) {
            A = vecA[i];
            B = vecB[i];
            mul += A * B;
            denomA += A * A;
            denomB += B * B;
        }
        if (denomA == 0 || denomB == 0) {
            throw std::logic_error("cosine similarity is not defined whenever one or both "
                                   "input vectors are zero-vectors.");
        }
        return mul / (sqrt(denomA) * sqrt(denomB));
    }

    std::shared_ptr<ov::Model> read(const ov::Core& core) override {
        // Read network model
        slog::info << "Reading model: " << FLAGS_m_reid << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m_reid);
        logBasicModelInfo(model);

        // set batch size 1
        model->get_parameters()[0]->set_layout("NCHW");
        ov::set_batch(model, 1);

        // Person Reidentification network should have 1 input and one output
        // Check inputs
        ov::OutputVector inputs = model->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Person Reidentification Retail should have 1 input");
        }

        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

        if (FLAGS_auto_resize) {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_spatial_dynamic_shape().
                set_layout({ "NHWC" });
            ppp.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW").
                resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            ppp.input().model().set_layout("NCHW");
        } else {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }

        model = ppp.build();

        m_inputName = model->input().get_any_name();

        // Check outputs
        ov::OutputVector outputs = model->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("Person Re-Identification Model should have 1 output");
        }

        m_outputName = model->output().get_any_name();

        m_enabled = true;

        return model;
    }
};
