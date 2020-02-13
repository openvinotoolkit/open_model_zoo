// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

