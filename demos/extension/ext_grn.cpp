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

#include <cmath>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class GRNImpl: public ExtLayerBase {
public:
    explicit GRNImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            bias = layer->GetParamAsFloat("bias");

            addConfig(layer, {{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

#if _MSC_VER && !__INTEL_COMPILER
        #pragma omp parallel for schedule(static)
#else
        #pragma omp parallel for collapse(3) schedule(static)
#endif
        for (int b = 0; b < N; b++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    double variance = 0;
                    for (int c = 0; c < C; c++) {
                        variance += std::pow(src_data[b*C*H*W + c*H*W + h*W + w], 2);
                    }
                    variance = std::pow(variance + bias, 0.5f);
                    for (int c = 0; c < C; c++) {
                        dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] / variance;
                    }
                }
            }
        }
        return OK;
    }

private:
    float bias = 1.0f;
};

REG_FACTORY_FOR(ImplFactory<GRNImpl>, GRN);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
