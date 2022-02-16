// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>

#include "text_recognition.hpp"

bool fileExists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}
bool tryReadVocabFile(const std::string& filename, std::string& alphabet) {
    if (fileExists(filename)) {
        alphabet = "";
        std::ifstream inputFile(filename);
        if (!inputFile.is_open()) {
            throw std::runtime_error("Can't open the vocab file: " + filename);
        }
        std::string vocabLine;
        while (std::getline(inputFile, vocabLine)) {
            if (vocabLine.length() != 1) {
                throw std::invalid_argument("Lines in the vocab file must contain 1 character");
            }
            alphabet += vocabLine;
        }
        if (alphabet.empty()) {
            throw std::logic_error("File is empty: " + filename);
        }
        return true;
    } else {
        return false;
    }
}

void softmaxAndChoose(const std::vector<float>::const_iterator& begin,
                      const std::vector<float>::const_iterator& end,
                      int *argmax, float *prob) {
    auto maxElem = std::max_element(begin, end);
    *argmax = static_cast<int>(std::distance(begin, maxElem));
    float maxVal = *maxElem;
    double sum = 0;
    for (auto i = begin; i != end; i++) {
        sum += std::exp((*i) - maxVal);
    }
    if (std::fabs(sum) < std::numeric_limits<double>::epsilon()) {
        throw std::logic_error("sum can't be equal to zero");
    }
    *prob = 1.0f / static_cast<float>(sum);
}

std::vector<float> softmax(const std::vector<float>::const_iterator& begin,
                           const std::vector<float>::const_iterator& end) {
    std::vector<float> prob(end - begin, 0.f);
    std::transform(begin, end, prob.begin(), [](float x) { return std::exp(x); });
    float sum = std::accumulate(prob.begin(), prob.end(), 0.0f);
    for (int i = 0; i < static_cast<int>(prob.size()); i++)
        prob[i] /= sum;
    return prob;
}

struct BeamElement {
    std::vector<int> sentence;  //!< The sequence of chars that will be a result of the beam element
    float probBlank;            //!< The probability that the last char in CTC sequence
                                //!< for the beam element is the special blank char
    float probNotBlank;         //!< The probability that the last char in CTC sequence
                                //!< for the beam element is NOT the special blank char

    float prob() const {        //!< The probability of the beam element.
        return probBlank + probNotBlank;
    }
};

std::string SimpleDecoder(const std::vector<float>& data, const std::string& alphabet,
                          char padSymbol, double *conf, int startIdx) {
    std::string result = "";
    const int numClasses = alphabet.length();
    *conf = 1;

    for (auto it = data.begin() + startIdx * numClasses; it != data.end(); it += numClasses) {
        int argmax;
        float prob;

        softmaxAndChoose(it, it + numClasses, &argmax, &prob);
        (*conf) *= prob;
        auto symbol = alphabet[argmax];
        if (symbol != padSymbol)
            result += symbol;
        else
            break;
    }

    return result;
}

std::string CTCGreedyDecoder(const std::vector<float>& data, const std::string& alphabet,
                             char padSymbol, double *conf) {
    std::string result = "";
    bool padPrev = false;
    *conf = 1;

    const int numClasses = alphabet.length();
    for (auto it = data.begin(); it != data.end(); it += numClasses) {
      int argmax;
      float prob;

      softmaxAndChoose(it, it + numClasses, &argmax, &prob);

      (*conf) *= prob;

      auto symbol = alphabet[argmax];
      if (symbol != padSymbol) {
          if (result.empty() || padPrev || (!result.empty() && symbol != result.back())) {
            padPrev = false;
            result += symbol;
          }
      } else {
        padPrev = true;
      }
    }

    return result;
}

std::string CTCBeamSearchDecoder(const std::vector<float>& data, const std::string& alphabet,
                                 char padSymbol, double *conf, int bandwidth) {
    const int numClasses = alphabet.length();

    std::vector<BeamElement> curr;
    std::vector<BeamElement> last;

    last.push_back(BeamElement{std::vector<int>(), 1.f, 0.f});

    for (auto it = data.begin(); it != data.end(); it += numClasses) {
        curr.clear();

        std::vector<float> prob = softmax(it, it + numClasses);

        for(const auto& candidate: last) {
            float probNotBlank = 0.f;
            const std::vector<int>& candidateSentence = candidate.sentence;
            if (!candidateSentence.empty()) {
                int n = candidateSentence.back();
                probNotBlank = candidate.probNotBlank * prob[n];
            }
            float probBlank = candidate.prob() * prob[numClasses - 1];

            auto cmp = [&candidateSentence](const BeamElement& n) {
                return n.sentence == candidateSentence;
            };
            auto checkRes = std::find_if(curr.begin(), curr.end(), cmp);
            if (checkRes == std::end(curr)) {
                curr.push_back(BeamElement{candidate.sentence, probBlank, probNotBlank});
            } else {
                checkRes->probNotBlank  += probNotBlank;
                if (checkRes->probBlank != 0.f) {
                    throw std::logic_error("Probability that the last char in CTC-sequence "
                                           "is the special blank char must be zero here");
                }
                checkRes->probBlank = probBlank;
            }

            for (int i = 0; i < numClasses - 1; i++) {
                auto extend = candidateSentence;
                extend.push_back(i);

                if (candidateSentence.size() > 0 && candidate.sentence.back() == i) {
                    probNotBlank = prob[i] * candidate.probBlank;
                } else {
                    probNotBlank = prob[i] * candidate.prob();
                }

                auto checkRes = std::find_if(
                    curr.begin(), curr.end(),
                    [&extend](const BeamElement& n) {
                        return n.sentence == extend;
                });

                if (checkRes == std::end(curr)) {
                    curr.push_back(BeamElement{extend, 0.f, probNotBlank});
                } else {
                    checkRes->probNotBlank += probNotBlank;
                }
            }
        }

        sort(curr.begin(), curr.end(), [](const BeamElement& a, const BeamElement& b) -> bool {
            return a.prob() > b.prob();
        });

        last.clear();
        int numToCopy = std::min(bandwidth, static_cast<int>(curr.size()));
        for (int b = 0; b < numToCopy; b++) {
            last.push_back(curr[b]);
        }
    }

    *conf = last[0].prob();
    std::string result = "";
    for (const auto& idx: last[0].sentence) {
        result += alphabet[idx];
    }

    return result;
}
