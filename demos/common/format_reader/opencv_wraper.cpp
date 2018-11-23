/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#ifdef USE_OPENCV
#include "opencv_wraper.h"
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <../samples/slog.hpp>

using namespace std;
using namespace FormatReader;

OCVReader::OCVReader(const string &filename) {
    img = cv::imread(filename);
    _size = 0;

    if (img.empty()) {
        return;
    }

    _size   = img.size().width * img.size().height * img.channels();
    _width  = img.size().width;
    _height = img.size().height;
}

std::shared_ptr<unsigned char> OCVReader::getData(int width = 0, int height = 0) {
    cv::Mat resized(img);
    if (width != 0 && height != 0) {
        int iw = img.size().width;
        int ih = img.size().height;
        if (width != iw || height != ih) {
            slog::warn << "Image is resized from (" << iw << ", " << ih << ") to (" << width << ", " << height << ")" << slog::endl;
        }
        cv::resize(img, resized, cv::Size(width, height));
    }

    size_t size = resized.size().width * resized.size().height * resized.channels();
    _data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
    for (size_t id = 0; id < size; ++id) {
        _data.get()[id] = resized.data[id];
    }
    return _data;
}
#endif
