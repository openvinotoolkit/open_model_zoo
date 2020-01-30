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
#include <numeric>

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

    std::vector<float> softmax(const std::vector<float>::const_iterator& begin, const std::vector<float>::const_iterator& end) {
        std::vector<float> prob(end - begin, 0.f);
        std::transform(begin, end, prob.begin(), [](float x) { return std::exp(x); });
        float sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
        for (int i = 0; i < static_cast<int>(prob.size()); i++)
            prob[i] /= sum;
        return prob;
    }

    struct BeamElement {
        std::vector<int> sentence;   //!< The sequence of chars that will be a result of the beam element
        float prob_blank;            //!< The probability that the last char in CTC sequence
                                     //!< for the beam element is the special blank char
        float prob_not_blank;        //!< The probability that the last char in CTC sequence
                                     //!< for the beam element is NOT the special blank char

        float prob() const {         //!< The probability of the beam element.
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
    const int num_classes = alphabet.length();

    std::vector<BeamElement> curr;
    std::vector<BeamElement> last;

    last.push_back(BeamElement{std::vector<int>(), 1.f, 0.f});

    for (std::vector<float>::const_iterator it = data.begin(); it != data.end(); it += num_classes) {
        curr.clear();

        std::vector<float> prob = softmax(it, it + num_classes);

        for(const auto& candidate: last) {
            float prob_not_blank = 0.f;
            const std::vector<int>& candidate_sentence = candidate.sentence;
            if (!candidate_sentence.empty()) {
                int n = candidate_sentence.back();
                prob_not_blank = candidate.prob_not_blank * prob[n];
            }
            float prob_blank = candidate.prob() * prob[num_classes - 1];

            auto check_res = std::find_if(curr.begin(), curr.end(), [&candidate_sentence](const BeamElement& n) {
                return n.sentence == candidate_sentence;
            });
            if (check_res == std::end(curr)) {
                curr.push_back(BeamElement{candidate.sentence, prob_blank, prob_not_blank});
            } else {
                check_res->prob_not_blank  += prob_not_blank;
                if (check_res->prob_blank != 0.f) {
                    throw std::logic_error("Probability that the last char in CTC-sequence is the special blank char must be zero here");
                }
                check_res->prob_blank = prob_blank;
            }

            for (int i = 0; i < num_classes - 1; i++) {
                auto extend = candidate_sentence;
                extend.push_back(i);

                if (candidate_sentence.size() > 0 && candidate.sentence.back() == i) {
                    prob_not_blank = prob[i] * candidate.prob_blank;
                } else {
                    prob_not_blank = prob[i] * candidate.prob();
                }
                
                auto check_res = std::find_if(curr.begin(), curr.end(), [&extend](const BeamElement &n) {
                    return n.sentence == extend;
                });

                if (check_res == std::end(curr)) {
                    curr.push_back(BeamElement{extend, 0.f, prob_not_blank});
                } else {
                    check_res->prob_not_blank += prob_not_blank;
                }
            }
        }

        sort(curr.begin(), curr.end(), [](const BeamElement &a, const BeamElement &b) -> bool {
            return a.prob() > b.prob();
        });

        last.clear();
        int num_to_copy = std::min(bandwidth, static_cast<int>(curr.size()));
        for (int b = 0; b < num_to_copy; b++) {
            last.push_back(curr[b]);
        }
    }

    *conf = last[0].prob();
    std::string res="";
    for (const auto& idx: last[0].sentence) {
        res += alphabet[idx];
    }

    return res;
}
