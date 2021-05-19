/*
// Copyright (C) 2018-2021 Intel Corporation
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
#include <va/va.h>

template <int a, int b, int c, int d>
struct FourccVal
{
    enum { code = (a) | (b << 8) | (c << 16) | (d << 24) };
};

enum FourCC {
    FOURCC_NONE = 0,
    FOURCC_RGBP_F32 = 0x07282024,
    FOURCC_NV12 = FourccVal<'N', 'V', '1', '2'>::code,
    FOURCC_BGRA = FourccVal<'B', 'G', 'R', 'A'>::code,
    FOURCC_BGRX = FourccVal<'B', 'G', 'R', 'X'>::code,
    FOURCC_BGRP = FourccVal<'B', 'G', 'R', 'P'>::code,
    FOURCC_BGR = FourccVal<'B', 'G', 'R', ' '>::code,
    FOURCC_RGBA = FourccVal<'R', 'G', 'B', 'A'>::code,
    FOURCC_RGBX = FourccVal<'R', 'G', 'B', 'X'>::code,
    FOURCC_RGB = FourccVal<'R', 'G', 'B', ' '>::code,
    FOURCC_RGBP = FourccVal<'R', 'G', 'B', 'P'>::code,
    FOURCC_I420 = FourccVal<'I', '4', '2', '0'>::code
};

int fourCCToVART(FourCC fourcc);
