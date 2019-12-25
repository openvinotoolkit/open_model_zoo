// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with ouput classification results
 * @file classification_results.hpp
 */
#include <string>
#include <vector>
#include <iostream>
#include <utility>

#include <ie_blob.h>

/**
 * @class ClassificationResult
 * @brief A ClassificationResult creates an output table with results
 */
class ClassificationResult {
private:
    const std::string _classidStr = "classid";
    const std::string _probabilityStr = "probability";
    const std::string _labelStr = "label";
    size_t _nTop;
    InferenceEngine::Blob::Ptr _outBlob;
    const std::vector<std::string> _labels;
    const std::vector<std::string> _imageNames;
    const size_t _batchSize;

    void printHeader() {
        std::cout << _classidStr << " " << _probabilityStr;
        if (!_labels.empty())
            std::cout << " " << _labelStr;
        std::string classidColumn(_classidStr.length(), '-');
        std::string probabilityColumn(_probabilityStr.length(), '-');
        std::string labelColumn(_labelStr.length(), '-');
        std::cout << std::endl << classidColumn << " " << probabilityColumn;
        if (!_labels.empty())
            std::cout << " " << labelColumn;
        std::cout << std::endl;
    }

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
        // if (_imageNames.size() != _batchSize) {
        //     throw std::logic_error("Batch size should be equal to the number of images.");
        // }
    }

    /**
    * @brief prints formatted classification results
    */
    void print() {
        /** This vector stores id's of top N results **/
        std::vector<unsigned> results = topResults(_nTop, *_outBlob);

        /** Print the result iterating over each batch **/
        std::cout << std::endl << "Top " << _nTop << " results:" << std::endl << std::endl;
        for (unsigned int image_id = 0; image_id < _batchSize; ++image_id) {
            std::cout << "Image " << _imageNames[image_id] << std::endl << std::endl;
            printHeader();

            for (size_t id = image_id * _nTop, cnt = 0; id < (image_id + 1) * _nTop; ++cnt, ++id) {
                std::cout.precision(7);
                /** Getting probability for resulting class **/
                const auto result = _outBlob->buffer().
                        as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>()
                [results[id] + image_id * (_outBlob->size() / _batchSize)];

                std::cout << std::setw(static_cast<int>(_classidStr.length())) << std::left << results[id] << " ";
                std::cout << std::left << std::setw(static_cast<int>(_probabilityStr.length())) << std::fixed << result;

                if (!_labels.empty()) {
                    std::cout << " " + _labels[results[id]];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
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
};
