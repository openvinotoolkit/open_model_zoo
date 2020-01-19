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
    void softmax_and_choose(const std::vector<float>::const_iterator& begin, const std::vector<float>::const_iterator& end, int *argmax, float *prob) {
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

    void softmax(const std::vector<float>::const_iterator& begin, const std::vector<float>::const_iterator& end, std::vector<float> &prob)
    {
        std::transform(begin, end, prob.begin(), static_cast<double(*)(double)>(std::exp));
        float sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
        for (int i = 0; i < static_cast<int>(prob.size()); i++)
            prob[i] /= sum;
    }

    struct BeamElement
    {
        std::string sentence;   //!< The sequence of chars that will be result of the beam element
        float prob_blank;       //!< The probability that the last char in CTC sequence
                                //!< for the beam element is the special blank char
        float prob_not_blank;   //!< The probability that the last char in CTC sequence
                                //!<  for the beam element is NOT the special blank char

        float prob()            //!< The probability of the beam element.
        {
            return prob_blank + prob_not_blank;
        }
    };

}  // namespace

std::string CTCGreedyDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double *conf) {
    std::string res = "";
    bool prev_pad = false;
    *conf = 1;

    const int num_classes = alphabet.length();
    for (std::vector<float>::const_iterator it = data.begin(); it != data.end(); it += num_classes) {
      int argmax;
      float prob;

      softmax_and_choose(it, it + num_classes, &argmax, &prob);

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

    std::vector<BeamElement> curr;
    std::vector<BeamElement> last;

    last.push_back(BeamElement{"", 1.f, 0.f});

    for (std::vector<float>::const_iterator it = data.begin(); it != data.end(); it += num_classes) {
        curr.clear();

        std::vector<float> prob = std::vector<float>(num_classes, 0.f);
        softmax(it, it + num_classes, prob);

        for (int candidate_num = 0; candidate_num < static_cast<int>(last.size()); candidate_num++) {
            float prob_not_blank = 0.f;
            auto candidate = last[candidate_num];
            std::string candidate_sentence = candidate.sentence;
            if (candidate_sentence != "") {
                int n = static_cast<int>(candidate_sentence.back());
                prob_not_blank = candidate.prob_blank * prob[n];
            }
            float prob_blank = candidate.prob() * prob[(num_classes - 1)];

            auto check_res = std::find_if(curr.begin(), curr.end(), [candidate_sentence](const BeamElement& n) {
                return n.sentence == candidate_sentence;
            });
            if (check_res == std::end(curr)) {
                curr.push_back(BeamElement{candidate.sentence, prob_blank, prob_not_blank});
            } else {
                auto index = std::distance(curr.begin(), check_res);
                curr[index].prob_not_blank  += prob_not_blank;
                curr[index].prob_blank = prob_blank;
            }

            for (int i = 0; i < num_classes - 1; i++) {
                auto extend = candidate_sentence + static_cast<char>(i);
                if (candidate_sentence.length() > 0 && candidate.sentence.back() == static_cast<char>(i)) {
                    prob_not_blank = prob[i] * candidate.prob_blank;
                } else {
                    prob_not_blank = prob[i] * candidate.prob();
                }
                
                auto check_res = std::find_if(curr.begin(), curr.end(), [extend](const BeamElement &n) {
                    return n.sentence == extend;
                });

                if (check_res == std::end(curr)) {
                    curr.push_back(BeamElement{extend, 0.f, prob_not_blank});
                } else {
                    auto index = std::distance(curr.begin(), check_res);
                    curr[index].prob_not_blank += prob_not_blank;
                }
            }
        }

        sort(curr.begin(), curr.end(), [](BeamElement &a, BeamElement &b) -> bool {
            return a.prob() > b.prob();
        });

        last.clear();
        int num_to_copy = std::min(bandwidth, static_cast<int>(curr.size()));
        for (int b = 0; b < num_to_copy; b++) {
            last.push_back(curr[b]);
        }
    }

    auto idx = last[0].sentence;
    *conf = last[0].prob();
    for (int _idx = 0; _idx < static_cast<int>(idx.length()); _idx++)
    {
        res += alphabet[static_cast<int>(idx[_idx])];
    }

    return res;
}

