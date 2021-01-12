/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

// After updating this file or its dependencies, please run
//   swig -python -c++ -o decoders_wrap.cpp decoders.i
// to update decoders_wrap.cpp and impl.py

%module impl

%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%{
#include "scorer_base.h"
#include "scorer_yoklm.h"
#include "binding.h"
%}

%include "std_vector.i"
%include "std_pair.i"
%include "std_string.i"

// Add support for size_t to numpy.i
%numpy_typemaps(int   , NPY_INT   , size_t)
%numpy_typemaps(float , NPY_FLOAT , size_t)

namespace std {
    %template(IntVector) std::vector<int>;
    %template(StringVector) std::vector<std::string>;
}

%apply (float * IN_ARRAY3, size_t DIM1, size_t DIM2, size_t DIM3) {(const float * probs, size_t batch_size, size_t max_frames, size_t num_classes)}
%apply (int * IN_ARRAY1, size_t DIM1) {(const int * seq_lens, size_t seq_lens_dim_batch)}
%apply (int ** ARGOUTVIEWM_ARRAY1, size_t * DIM1) {(int ** tokens, size_t * tokens_dim)}
%apply (int ** ARGOUTVIEWM_ARRAY1, size_t * DIM1) {(int ** timesteps, size_t * timesteps_dim)}
%apply (float ** ARGOUTVIEWM_ARRAY1, size_t * DIM1) {(float ** scores, size_t * scores_dim)}
%apply (int ** ARGOUTVIEWM_ARRAY1, size_t * DIM1) {(int ** tokens_lengths, size_t * tokens_lengths_dim)}

// Workaround for the absent support of std::unique_ptr<...>.
%ignore ScorerBase::dictionary;

%include "scorer_base.h"
%include "scorer_yoklm.h"
%include "binding.h"
