/*
// Copyright (c) 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class ReorgYoloImpl: public ExtLayerBase {
public:
    explicit ReorgYoloImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            stride = layer->GetParamAsInt("stride");

            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();

        int IW = (inputs[0]->getTensorDesc().getDims().size() > 3) ? inputs[0]->getTensorDesc().getDims()[3] : 1;
        int IH = (inputs[0]->getTensorDesc().getDims().size() > 2) ? inputs[0]->getTensorDesc().getDims()[2] : 1;
        int IC = (inputs[0]->getTensorDesc().getDims().size() > 1) ? inputs[0]->getTensorDesc().getDims()[1] : 1;
        int B = (inputs[0]->getTensorDesc().getDims().size() > 0) ? inputs[0]->getTensorDesc().getDims()[0] : 1;

        int ic_off = IC / (stride * stride);
        int ih_off = IH * stride;
        int iw_off = IW * stride;
        for (int b = 0; b < B; b++) {
            for (int ic = 0; ic < IC; ic++) {
                for (int ih = 0; ih < IH; ih++) {
                    for (int iw = 0; iw < IW; iw++) {
                        int dstIndex = b * IC * IH * IW + ic * IH * IW + ih * IW + iw;

                        int oc = ic % ic_off;
                        int offset = ic / ic_off;

                        int ow = iw * stride + offset % stride;
                        int oh = ih * stride + offset / stride;

                        int srcIndex = b * ic_off * ih_off * iw_off + oc * ih_off * iw_off + oh * iw_off + ow;

                        dst_data[dstIndex] = src_data[srcIndex];
                    }
                }
            }
        }
        return OK;
    }

private:
    int stride;
};

REG_FACTORY_FOR(ImplFactory<ReorgYoloImpl>, ReorgYolo);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
