// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <array>

bool tryReadVocabFile(const std::string& filename, std::string& alphabet);

void checkCompositeNetNames(const std::vector<std::string>&  encOutputLayers,
                            const std::array<std::string,2>& encOutputNames,
                            const std::vector<std::string>&  decInputLayers,
                            const std::array<std::string,3>& decInputNames,
                            const std::vector<std::string>&  decOutputLayers,
                            const std::array<std::string,2>& decOutputNames);

std::string CTCGreedyDecoder(const std::vector<float> &data, const std::string& alphabet,
                             char padSymbol, double *conf);
std::string CTCBeamSearchDecoder(const std::vector<float> &data, const std::string& alphabet,
                                 char padSymbol, double *conf, int bandwidth);
std::string SimpleDecoder(const std::vector<float> &data, const std::string& alphabet,
                          char padSymbol, double *conf, int startIdx);
