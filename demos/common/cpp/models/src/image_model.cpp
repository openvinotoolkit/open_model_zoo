/*
// Copyright (C) 2021 Intel Corporation
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

#include "models/image_model.h"
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>
#include <utils/uni_image.h>
#include <cldnn/cldnn_config.hpp>

#ifdef USE_VA
#include <gpu/gpu_context_api_va.hpp>
#include <ie_compound_blob.h>

#include "gst_vaapi_decoder.h"
#endif

using namespace InferenceEngine;

ImageModel::ImageModel(const std::string& modelFileName, bool useAutoResize) :
    ModelBase(modelFileName),
    useAutoResize(useAutoResize) {
}

InferenceEngine::ExecutableNetwork ImageModel::loadExecutableNetwork(const CnnConfig & cnnConfig, InferenceEngine::Core & core) {
    this->cnnConfig = cnnConfig;
    auto cnnNetwork = prepareNetwork(core);

    if(cnnConfig.remoteContext)
    {
        // Here we adjust configuration and loading the networkwith respect to provided remote context
        // Setting image input (0-index input is image input) to use NV12
        InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
        inputInfo[inputsNames[0]]->getPreProcess().setColorFormat(ColorFormat::NV12);

        // Loading network
        auto cfg = cnnConfig.execNetworkConfig;

        // TODO: Remove this workaround after the problem is fixed
        if(cfg.find(CONFIG_KEY(GPU_THROUGHPUT_STREAMS))!= cfg.end()) {
            slog::warn << "GPU Remote context mode does not work with nstreams>1. Number of streams was reset to 1." << slog::endl;
            cfg[CONFIG_KEY(GPU_THROUGHPUT_STREAMS)] = "1";
        }

        cfg[CLDNNConfigParams::KEY_CLDNN_NV12_TWO_INPUTS] = PluginConfigParams::YES;
        execNetwork = core.LoadNetwork(cnnNetwork, cnnConfig.remoteContext, cfg);
    }
    else
        execNetwork = core.LoadNetwork(cnnNetwork, cnnConfig.devices, cnnConfig.execNetworkConfig);

    return execNetwork;
}

std::shared_ptr<InternalModelData> ImageModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto& data = inputData.asRef<ImageInputData>();
    auto& img = useAutoResize ? data.inputImage :
            data.inputImage->resize(netInputWidth, netInputHeight);

    /* Resize and copy data from the image to the input blob */
    request->SetBlob(inputsNames[0], img->toBlob(isNHWCModelInput));

    // Keeping image in internal data is important, as long as Blob shares data taken from Mat or other sources,
    // so if source would be destroyed before async processing is over, Blob will loose the data
    return std::make_shared<InternalImageModelData>(data.inputImage->size().width, data.inputImage->size().height, img);
}
