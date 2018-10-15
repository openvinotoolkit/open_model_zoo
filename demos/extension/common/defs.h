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
#pragma once

#if defined (HAVE_SSE) || defined (HAVE_AVX2)
#if defined (_WIN32)
#include <emmintrin.h>
#else
#include <x86intrin.h>
#endif
#endif

#if defined (WIN32) || defined (_WIN32)
#if defined (__INTEL_COMPILER)
#define DLSDK_EXT_IVDEP() __pragma(ivdep)
#elif defined(_MSC_VER)
#define DLSDK_EXT_IVDEP() __pragma(loop(ivdep))
#else
#define DLSDK_EXT_IVDEP()
#endif
#elif defined(__linux__)
#if defined(__INTEL_COMPILER)
#define DLSDK_EXT_IVDEP() _Pragma("ivdep")
#elif defined(__GNUC__)
#define DLSDK_EXT_IVDEP() _Pragma("GCC ivdep")
#else
#define DLSDK_EXT_IVDEP()
#endif
#else
#define DLSDK_EXT_IVDEP()
#endif