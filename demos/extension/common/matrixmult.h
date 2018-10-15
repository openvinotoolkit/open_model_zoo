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

static inline void matrixMult(float *A, float *B, float *C, int m, int n, int k, bool transposeB = false) {
    if (transposeB) {
        for (int rowA = 0; rowA < m; rowA++) {
            for (int rowB = 0; rowB < n; rowB++) {
                float sum = 0;
                for (int colA = 0; colA < k; colA++) {
                    sum += A[rowA * k + colA] * B[rowB * k + colA];
                }

                C[rowA * n + rowB] = sum;
            }
        }
    } else {
        for (int rowA = 0; rowA < m; rowA++) {
            for (int colB = 0; colB < n; colB++) {
                float sum = 0;
                for (int colA = 0; colA < k; colA++) {
                    sum += A[rowA * k + colA] * B[colA * n + colB];
                }

                C[rowA * n + colB] = sum;
            }
        }
    }
}