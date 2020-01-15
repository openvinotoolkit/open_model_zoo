// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

class beam
{
public:
    beam(std::string _text, float _pB, float _pNB, float _pT)
    {
        sentance = _text;
        pB = _pB;
        pNB = _pNB;
        pT = _pT;
    };
public :
    std::string sentance;
    float pB;
    float pNB;
    float pT;
};

std::string CTCGreedyDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double *conf);
std::string CTCBeamSearchDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double *conf, int bandwidth);
