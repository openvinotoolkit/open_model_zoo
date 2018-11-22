/*
// Copyright (c) 2018 Intel Corporation
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

#include "core.hpp"

#include <iostream>

bool operator==(const TrackedObject& first, const TrackedObject& second) {
    return ( (first.rect == second.rect)
            && (first.confidence == second.confidence)
            && (first.frame_idx == second.frame_idx)
            && (first.object_id == second.object_id)
            && (first.timestamp == second.timestamp) );
}

bool operator!=(const TrackedObject& first, const TrackedObject& second) {
    return !(first == second);
}
