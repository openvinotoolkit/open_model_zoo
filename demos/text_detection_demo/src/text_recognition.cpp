// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "text_recognition.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
#include <stdexcept>

namespace  {
    void softmax(const std::vector<float>::const_iterator& begin, const std::vector<float>::const_iterator& end, int *argmax, float *prob) {
        auto max_element = std::max_element(begin, end);
        *argmax = static_cast<int>(std::distance(begin, max_element));
        float max_val = *max_element;
        double sum = 0;
        for (auto i = begin; i != end; i++) {
           sum += std::exp((*i) - max_val);
        }
        if (std::fabs(sum) < std::numeric_limits<double>::epsilon()) {
            throw std::logic_error("sum can't be equal to zero");
        }
        *prob = 1.0f / static_cast<float>(sum);
    }
}  // namespace

std::string CTCGreedyDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double *conf) {
    std::string res = "";
    bool prev_pad = false;
    *conf = 1;

    const int num_classes = alphabet.length();
    for (std::vector<float>::const_iterator it = data.begin(); it != data.end(); it += num_classes) {
      int argmax;
      float prob;

      softmax(it, it + num_classes, &argmax, &prob);

      (*conf) *= prob;

      auto symbol = alphabet[argmax];
      if (symbol != pad_symbol) {
          if (res.empty() || prev_pad || (!res.empty() && symbol != res.back())) {
            prev_pad = false;
            res += symbol;
          }
      } else {
        prev_pad = true;
      }
    }
    return res;
}
