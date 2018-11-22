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

#include "threading.hpp"

#ifdef USE_TBB
#include <cassert>

namespace {
tbb::task_arena *arena_ptr = nullptr;
}  // namespace
void TbbArenaWrapper::init() {
    assert(nullptr == arena_ptr);
    arena_ptr = &arena;
}

void TbbArenaWrapper::deinit() {
    assert(&arena == arena_ptr);
    arena_ptr = nullptr;
}

tbb::task_arena& get_tbb_arena() {
    assert(nullptr != arena_ptr);
    return *arena_ptr;
}
#endif

