/*********************************************************************
* Copyright (c) 2020-2024 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

// After updating this file or its dependencies, please run
//   swig -python -c++ -o decoders_wrap.cpp decoders.i
// to update decoders_wrap.cpp and impl.py

%module impl

// Catch C++ exceptions to ensure they're reported by Python
%include "exception.i"
%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

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
#include "ctc_beam_search_decoder.h"
#include "binding.h"
%}

%include "std_vector.i"
%include "std_string.i"

// Add support for size_t to numpy.i
%numpy_typemaps(int   , NPY_INT   , size_t)
%numpy_typemaps(float , NPY_FLOAT , size_t)

namespace std {
    %template(StringVector) std::vector<std::string>;
}

%apply (float * IN_ARRAY3, size_t DIM1, size_t DIM2, size_t DIM3) {(const float * probs, size_t batch_size, size_t max_frames, size_t num_classes)}
%apply (float * IN_ARRAY2, size_t DIM1, size_t DIM2) {(const float * probs, size_t num_frames, size_t num_classes)}
%apply (int * IN_ARRAY1, size_t DIM1) {(const int * seq_lens, size_t seq_lens_dim_batch)}
%apply (int ** ARGOUTVIEWM_ARRAY1, size_t * DIM1) {(int ** symbols, size_t * symbols_dim)}
%apply (int ** ARGOUTVIEWM_ARRAY1, size_t * DIM1) {(int ** timesteps, size_t * timesteps_dim)}
%apply (float ** ARGOUTVIEWM_ARRAY1, size_t * DIM1) {(float ** scores, size_t * scores_dim)}
%apply (int ** ARGOUTVIEWM_ARRAY1, size_t * DIM1) {(int ** symbols_lengths, size_t * symbols_lengths_dim)}

// Workaround for the absent support of std::unique_ptr<...>.
%ignore ScorerBase::dictionary;

%include "scorer_base.h"
%include "scorer_yoklm.h"
%include "ctc_beam_search_decoder.h"
%include "binding.h"
