// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <details/ie_exception.hpp>
#include <ie_layouts.h>
#include "ie_parallel.hpp"

using namespace TemplateExtension;

//! [cpu_implementation:ctor]
OpImplementation::OpImplementation(const std::shared_ptr<ngraph::Node> &node) {
    try {
        auto castedNode = std::dynamic_pointer_cast<Operation>(node);
        if (!castedNode)
            THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
        if (castedNode->inputs().size() != 4 || castedNode->outputs().size() != 1)
            THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
        if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
            THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
        if (castedNode->get_input_shape(0).size() != 4 || castedNode->get_output_shape(0).size() != 4)
            THROW_IE_EXCEPTION << "Operation supports only 4d tensors for input and output.";
        if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
            THROW_IE_EXCEPTION << "Operation supports only FP32 tensors.";
        inShapes.resize(4);
        for (int i = 0; i < 4; ++i)
            inShapes[i] = castedNode->get_input_shape(i);
        outShape = castedNode->get_output_shape(0);

        mask.resize(inShapes[1][0]*inShapes[1][1]*inShapes[1][2]*inShapes[1][3]);
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        error = ex.what();
    }
}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode OpImplementation::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                                         InferenceEngine::ResponseDesc *resp) noexcept {
     std::vector<InferenceEngine::DataConfig> inDataConfig;
     std::vector<InferenceEngine::DataConfig> outDataConfig;
     InferenceEngine::SizeVector order = {0, 1, 2, 3};
     // Allow any offset before data
     size_t offset((std::numeric_limits<size_t>::max)());

     // Input shape
     for (const auto& shape : inShapes)
     {
         InferenceEngine::DataConfig inpConf;
         inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, shape, {shape, order, offset});
         inDataConfig.push_back(inpConf);
     }

     // Output shape
     InferenceEngine::DataConfig outConf;
     outConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outShape, {outShape, order, offset});
     outDataConfig.push_back(outConf);

     InferenceEngine::LayerConfig layerConfig;
     layerConfig.inConfs = inDataConfig;
     layerConfig.outConfs = outDataConfig;

     conf.push_back(layerConfig);
     return InferenceEngine::StatusCode::OK;
}
//! [cpu_implementation:getSupportedConfigurations]

//! [cpu_implementation:init]
InferenceEngine::StatusCode OpImplementation::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        if (config.inConfs.size() != 4 || config.outConfs.size() != 1) {
            THROW_IE_EXCEPTION << "Operation cannot be initialized with incorrect number of inputs/outputs!";
        }

        if (config.inConfs[0].desc.getDims().size() != 4 || config.outConfs[0].desc.getDims().size() != 4) {
            THROW_IE_EXCEPTION << "Operation can be initialized only with 4d input/output tensors!";
        }

        if (config.outConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32 ||
                config.inConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32)  {
            THROW_IE_EXCEPTION << "Operation supports only FP32 precisions!";
        }
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        if (resp) {
            strncpy(resp->msg, error.c_str(), sizeof(resp->msg) - 1);
            resp->msg[sizeof(resp->msg)-1] = 0;
        }
        return InferenceEngine::GENERAL_ERROR;
    }

    return InferenceEngine::OK;
}
//! [cpu_implementation:init]

//! [cpu_implementation:execute]
InferenceEngine::StatusCode OpImplementation::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                                      std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                                      InferenceEngine::ResponseDesc *resp) noexcept {
    const float* poolInp = inputs[0]->cbuffer().as<float*>();
    const float* poolOut = inputs[1]->cbuffer().as<float*>();
    const float* inp     = inputs[2]->cbuffer().as<float*>();
    float* out = outputs[0]->buffer().as<float*>();

    std::vector<size_t> poolInpDims = inputs[0]->getTensorDesc().getDims();
    std::vector<size_t> poolOutDims = inputs[1]->getTensorDesc().getDims();
    std::vector<size_t> inpDims = inputs[2]->getTensorDesc().getDims();
    std::vector<size_t> outDims = outputs[0]->getTensorDesc().getDims();

    const size_t batch    = poolInpDims[0];
    const size_t channels = poolInpDims[1];
    const size_t height   = poolInpDims[2];
    const size_t width    = poolInpDims[3];
    const size_t outHeight = outDims[2];
    const size_t outWidth  = outDims[3];
    const size_t poolOutHeight = poolOutDims[2];
    const size_t poolOutWidth  = poolOutDims[3];
    std::fill(mask.begin(), mask.end(), false);
    memset(out, 0, outputs[0]->byteSize());
    InferenceEngine::parallel_for(batch*channels, [&](size_t d) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int poolOutIdx = (d * poolOutHeight + y / 2) * poolOutWidth + x / 2;
                int poolInpIdx = (d * height + y) * width + x;
                int dstIdx = d * outHeight * outWidth + (y * width + x);
                if (fabs(poolInp[poolInpIdx] - poolOut[poolOutIdx]) < 1e-5f && !mask[poolOutIdx]) {
                    out[dstIdx] = inp[poolOutIdx];
                    mask[poolOutIdx] = true;
                }
            }
        }
    });
    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]
