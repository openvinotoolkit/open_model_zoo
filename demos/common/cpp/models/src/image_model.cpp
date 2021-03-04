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

#ifdef USE_VA
#include <gpu/gpu_context_api_va.hpp>
#include <ie_compound_blob.h>
#include <cldnn/cldnn_config.hpp>

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

    // Preparing VA context and VA images pool
    if (cnnConfig.useGPURemoteContext) {
#ifdef USE_VA
        va_context.reset(new InferenceBackend::VaApiContext);
        va_converter.reset(new InferenceBackend::VaApiConverter(va_context.get()));

        sharedVAContext = InferenceEngine::gpu::make_shared_context(core, "GPU", va_context->Display());

        VaApiImagePool::ImageInfo info = { netInputWidth,netInputHeight,FOURCC_NV12,MemoryType::VAAPI };

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
        execNetwork = core.LoadNetwork(cnnNetwork, sharedVAContext, cfg);
        resizedSurfacesPool.reset(new InferenceBackend::VaApiImagePool(va_context.get(), cnnConfig.maxAsyncRequests+1, info));
#else
        throw std::runtime_error("Demos should be compiled with ENABLE_VA=TRUE option to use remote GPU context");
#endif
    }
    else
        execNetwork = core.LoadNetwork(cnnNetwork, cnnConfig.devices, cnnConfig.execNetworkConfig);
    return execNetwork;
}

std::shared_ptr<InternalModelData> ImageModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    auto& data = inputData.asRef<ImageInputData>();
    int width = 0;
    int height = 0;
    if(data.isVA())
    {
#ifdef USE_VA
        if(!cnnConfig.useGPURemoteContext)
        {
            throw std::runtime_error("Direct GPU copy was not initialized, but input data containing VA surface is received");
        }

        auto& vaImg = data.vaImage;
        width = vaImg->width;
        height = vaImg->height;

        auto resizedImg = resizedSurfacesPool->Acquire();
        va_converter->Convert(*vaImg, *resizedImg->image);

        auto nv12_blob = InferenceEngine::gpu::make_shared_blob_nv12(netInputHeight, netInputWidth, sharedVAContext, resizedImg->image->va_surface_id);

        request->SetBlob(inputsNames[0],nv12_blob);
        return std::shared_ptr<InternalModelData>(new InternalImageModelData(width, height,resizedImg));
#else
        throw std::runtime_error("Direct GPU copy was not initialized, but input data containing VA surface is received. You have to compile code with -ENABLE_VA option as well.");
#endif
        return std::shared_ptr<InternalModelData>(new InternalImageModelData(width, height));
    }
    else {
        auto& img = data.inputImage;
        width = img.cols;
        height = img.rows;

        if (useAutoResize) {
            /* Just set input blob containing read image. Resize and layout conversionx will be done automatically */
            request->SetBlob(inputsNames[0], wrapMat2Blob(img));
            /* IE::Blob::Ptr from wrapMat2Blob() doesn't own data. Save the image to avoid deallocation before inference */
            return std::make_shared<InternalImageMatModelData>(img);
        }
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = request->GetBlob(inputsNames[0]);
        matU8ToBlob<uint8_t>(img, frameBlob);
        return std::make_shared<InternalImageModelData>(img.cols, img.rows);
    }
}
