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

#include <string>
#include <map>
#include <memory>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

std::shared_ptr<ExtensionsHolder> CpuExtensions::GetExtensionsHolder() {
    static std::shared_ptr<ExtensionsHolder> localHolder;
    if (localHolder == nullptr) {
        localHolder = std::shared_ptr<ExtensionsHolder>(new ExtensionsHolder());
    }
    return localHolder;
}

void CpuExtensions::AddExt(std::string name, ext_factory factory) {
    GetExtensionsHolder()->list[name] = factory;
}

void CpuExtensions::AddShapeInferImpl(std::string name, const IShapeInferImpl::Ptr& impl) {
    GetExtensionsHolder()->si_list[name] = impl;
}

void CpuExtensions::GetVersion(const Version*& versionInfo) const noexcept {
    static Version ExtensionDescription = {
            { 1, 0 },    // extension API version
            "1.0",
            "ie-cpu-ext"  // extension description message
    };

    versionInfo = &ExtensionDescription;
}

StatusCode CpuExtensions::getShapeInferImpl(IShapeInferImpl::Ptr& impl, const char* type, ResponseDesc* resp) noexcept {
    auto& factories = CpuExtensions::GetExtensionsHolder()->si_list;
    if (factories.find(type) == factories.end()) {
        std::string errorMsg = std::string("Shape Infer Implementation for ") + type + " wasn't found!";
        errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        return NOT_FOUND;
    }
    impl = factories[type];
    return OK;
}

// Exported function
INFERENCE_EXTENSION_API(StatusCode) CreateExtension(IExtension*& ext, ResponseDesc* resp) noexcept {
    try {
        ext = new CpuExtensions();
        return OK;
    } catch (std::exception& ex) {
        if (resp) {
            std::string err = ((std::string)"Couldn't create extension: ") + ex.what();
            err.copy(resp->msg, 255);
        }
        return GENERAL_ERROR;
    }
}

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine

