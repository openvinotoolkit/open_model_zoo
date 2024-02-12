/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
*
* This file is based in part on output.h from https://github.com/parlance/ctcdecode,
* commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
**********************************************************************/

#ifndef OUTPUT_H_
#define OUTPUT_H_

#include <vector>

// Struct for a single candidate text in beam search output, containing
// the symbols ("tokens") based on the alphabet indices, and the timesteps
// for each symbol in the beam search output.
struct Output {
    std::vector<int> tokens, timesteps;
    float audio_score;
};

#endif  // OUTPUT_H_
