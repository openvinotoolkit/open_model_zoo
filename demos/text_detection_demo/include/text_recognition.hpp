// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

std::string CTCGreedyDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double *conf);
std::string CTCGreedyDecoder_utf8(const std::vector<float> &data, const std::vector<std::string>& dict, std::string pad_symbol, double *conf);
std::string CTCBeamSearchDecoder(const std::vector<float> &data, const std::string& alphabet, char pad_symbol, double *conf, int bandwidth);
