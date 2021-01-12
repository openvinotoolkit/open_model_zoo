// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
