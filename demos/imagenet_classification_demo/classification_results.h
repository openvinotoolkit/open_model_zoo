// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <iostream>
#include <utility>

#include <ie_blob.h>

class ClassificationResult {
public:
    explicit ClassificationResult(InferenceEngine::Blob::Ptr output_blob,
                                  size_t batch_size = 1,
                                  size_t num_of_top = 10,
                                  std::vector<std::string> image_names = {},
                                  std::vector<std::string> labels = {}) :
            _nTop(num_of_top),
            _outBlob(std::move(output_blob)),
            _labels(std::move(labels)),
            _imageNames(std::move(image_names)),
            _batchSize(batch_size) {
    }

    std::vector<unsigned> topResults(unsigned int n, InferenceEngine::Blob& input) {
        std::vector<unsigned> output;

        switch (input.getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32: {
                using currentBlobType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type;
                InferenceEngine::TBlob<currentBlobType>& tblob = dynamic_cast<
                                                                    InferenceEngine::TBlob<currentBlobType>&>(input);
                topResults(n, tblob, output);
                break;
            }
            case InferenceEngine::Precision::FP16: {
                using currentBlobType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type;
                InferenceEngine::TBlob<currentBlobType>& tblob = dynamic_cast<
                                                                    InferenceEngine::TBlob<currentBlobType>&>(input);
                topResults(n, tblob, output);
                break;
            }
            default:
                THROW_IE_EXCEPTION << "cannot locate blob for precision: " << input.getTensorDesc().getPrecision();
        }

        return output;
    }

private:
    size_t _nTop;
    InferenceEngine::Blob::Ptr _outBlob;
    const std::vector<std::string> _labels;
    const std::vector<std::string> _imageNames;
    const size_t _batchSize;

    template<class T>
    inline void topResults(unsigned int n, InferenceEngine::TBlob<T> &input, std::vector<unsigned> &output) {
        InferenceEngine::SizeVector dims = input.getTensorDesc().getDims();
        size_t input_rank = dims.size();
        if (!input_rank || !dims[0])
            THROW_IE_EXCEPTION << "Input blob has incorrect dimensions!";
        size_t batchSize = dims[0];
        std::vector<unsigned> indexes(input.size() / batchSize);

        n = static_cast<unsigned>(std::min<size_t>((size_t) n, input.size()));

        output.resize(n * batchSize);

        for (size_t i = 0; i < batchSize; i++) {
            size_t offset = i * (input.size() / batchSize);
            T *batchData = input.data();
            batchData += offset;

            std::iota(std::begin(indexes), std::end(indexes), 0);
            std::partial_sort(std::begin(indexes), std::begin(indexes) + n, std::end(indexes),
                              [&batchData](unsigned l, unsigned r) {
                                  return batchData[l] > batchData[r];
                              });
            for (unsigned j = 0; j < n; j++) {
                output.at(i * n + j) = indexes.at(j);
            }
        }
    }
};
