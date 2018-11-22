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

#pragma once

#ifdef USE_TBB
#include <utility>

#include <tbb/task_arena.h>
#ifdef TBB_TASK_ISOLATION
#include <tbb/parallel_for.h>
#endif

struct TbbArenaWrapper final{
    tbb::task_arena arena;

    template<typename... Args>
    explicit TbbArenaWrapper(Args... args):
        arena(std::forward<Args>(args)...) {
        init();
    }
    TbbArenaWrapper(const TbbArenaWrapper&) = delete;

    ~TbbArenaWrapper() {
        deinit();
    }

private:
    void init();
    void deinit();
};
tbb::task_arena& get_tbb_arena();

template<typename F>
void run_in_arena(F&& func) {
    auto &arena = get_tbb_arena();
    arena.execute([&](){
#ifdef TBB_TASK_ISOLATION
        // Workarond for tbb task isolation bug
        tbb::parallel_for<int>(0, 1, [&](int) {
            tbb::this_task_arena::isolate([&](){
                func();
            });
        });
#else
        func();
#endif
    });
}
#endif
