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

    void softmax_layer(const std::vector<float>::const_iterator& begin, const std::vector<float>::const_iterator& end, std::vector<float> &prob)
    {
        std::transform(begin, end, prob.begin(), static_cast<double(*)(double)>(std::exp));
        float sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
        for (int i = 0; i < static_cast<int>(prob.size()); i++)
            prob[i] /= sum;
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

std::string CTCBeamSearchDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double *conf, int bandwidth) {
    std::string res = "";

    const int num_classes = alphabet.length();

    std::vector<beam> curr;
    std::vector<beam> last;

    beam init("", 1.f, 0.f, 1.f);
    last.push_back(init);

    for (std::vector<float>::const_iterator it = data.begin(); it != data.end(); it += num_classes) {
        curr.clear();

        std::vector<float> prob = std::vector<float>(num_classes, 0.f);
        softmax_layer(it, it + num_classes, prob);

        for (int _candidate = 0; _candidate < static_cast<int>(last.size()); _candidate++) {
            float _pNB = 0.f;
            auto __can = last[_candidate];
            std::string __can_sentance = __can.sentance;
            if (__can_sentance != "") {
                int n = static_cast<int>(__can_sentance.back());
                _pNB = __can.pNB * prob[n];
            }
            float _pB = __can.pT * prob[(num_classes - 1)];

            auto check_res = std::find_if(curr.begin(), curr.end(), [__can_sentance](beam const& n) {
                return n.sentance == __can_sentance;
            });
            if (check_res == std::end(curr)) {
                curr.push_back(beam(__can.sentance, _pB, _pNB, _pB + _pNB));
            } else {
                auto __i = std::distance(curr.begin(), check_res);
                curr[__i].pNB += _pNB;
                curr[__i].pB = _pB;
                curr[__i].pT = curr[__i].pB + curr[__i].pNB;
            }

            for (int i = 0; i < num_classes - 1; i++) {
                auto extand_t = __can_sentance + static_cast<char>(i);
                if (__can_sentance.length() > 0 && __can.sentance.back() == static_cast<char>(i)) {
                    _pNB = prob[i] * __can.pB;
                } else {
                    _pNB = prob[i] * __can.pT;
                }
                
                auto check_res = std::find_if(curr.begin(), curr.end(), [extand_t](beam const& n) {
                    return n.sentance == extand_t;
                });

                if (check_res == std::end(curr)) {
                    curr.push_back(beam(extand_t, 0.f, _pNB, _pNB));
                } else {
                    auto __i = std::distance(curr.begin(), check_res);
                    curr[__i].pNB += _pNB;
                    curr[__i].pT += _pNB;
                }
            }
        }

        sort(curr.begin(), curr.end(), [](const beam &a, const beam &b) -> bool {
            return a.pT > b.pT;
        });

        last.clear();
        if (bandwidth > static_cast<int>(curr.size())) {
            for (int _b = 0; _b < static_cast<int>(curr.size()); _b++) {
                last.push_back(curr[_b]);
            }
        } else {
            for (int _b = 0; _b < bandwidth; _b++) {
                last.push_back(curr[_b]);
            }
        }
    }

    auto idx = last[0].sentance;
    *conf = last[0].pT;
    for (int _idx = 0; _idx < static_cast<int>(idx.length()); _idx++)
    {
        res += alphabet[static_cast<int>(idx[_idx])];
    }

    return res;
}

