// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>

struct Captures {
    std::vector<cv::Mat>& inputImgs;//??
    std::string inputBlobName;
    int curPos;
    int batchSize;//?
    int framesNum;
    bool quitFlag;

    Captures(std::vector<cv::Mat>& inputImgs, std::string inputBlobName, int batchSize):
                     inputImgs(inputImgs),
                     inputBlobName(inputBlobName),
                     curPos(0),
                     batchSize(batchSize),
                     framesNum(0),
                     quitFlag(false) {}
};
