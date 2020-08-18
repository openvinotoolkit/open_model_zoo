/*
// Copyright (C) 2018-2020 Intel Corporation
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

#include "requests_pool.h"

RequestsPool::RequestsPool(InferenceEngine::ExecutableNetwork & execNetwork, unsigned int size)
{
    for (unsigned int infReqId = 0; infReqId < size; ++infReqId) {
        requestsPool.emplace(execNetwork.CreateInferRequestPtr(), false);
    }
}

InferenceEngine::InferRequest::Ptr RequestsPool::getIdleRequest()
{
    const auto& it = std::find_if(requestsPool.begin(), requestsPool.end(), [](std::pair<const InferenceEngine::InferRequest::Ptr, std::atomic_bool>& x) {return !x.second; });
    if (it == requestsPool.end())
    {
        return InferenceEngine::InferRequest::Ptr();
    }
    else
    {
        it->second = true;
        return it->first;
    }
}

void RequestsPool::waitForTotalCompletion() {
    for (const auto& pair : requestsPool) {
        if (pair.second)
            pair.first->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY);
    }
}
