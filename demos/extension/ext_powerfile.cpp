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

class PowerFileImpl: public ExtLayerBase {
public:
    explicit PowerFileImpl(const CNNLayer* layer) {
        try {
            if (layer->insData.size() != 1 || layer->outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            // TODO: load this from some file or as blob?
            shift_.push_back(1);
            shift_.push_back(0);
            shift_.push_back(0);
            shift_.push_back(0);
            shift_.push_back(1);
            shift_.push_back(0);

            addConfig(layer, {DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        if (inputs.size() != 1 || outputs.empty()) {
            if (resp) {
                std::string errorMsg = "Incorrect number of input or output edges!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        for (size_t i = 0; i < inputs[0]->size(); i++) {
            size_t shift_idx = i % shift_.size();
            dst_data[i] = src_data[i] + shift_[shift_idx];
        }
        return OK;
    }

private:
    std::vector<int> shift_;
};

REG_FACTORY_FOR(ImplFactory<PowerFileImpl>, PowerFile);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
