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

#include <iostream>
#include <format_reader.h>
#include "bmp.h"
#include "MnistUbyte.h"
#include "opencv_wraper.h"

using namespace FormatReader;

std::vector<Registry::CreatorFunction> Registry::_data;

Register<MnistUbyte> MnistUbyte::reg;
#ifdef USE_OPENCV
Register<OCVReader> OCVReader::reg;
#else
Register<BitMap> BitMap::reg;
#endif

Reader *Registry::CreateReader(const char *filename) {
    for (auto maker : _data) {
        Reader *ol = maker(filename);
        if (ol != nullptr && ol->size() != 0) return ol;
        if (ol != nullptr) ol->Release();
    }
    return nullptr;
}

void Registry::RegisterReader(CreatorFunction f) {
    _data.push_back(f);
}

FORMAT_READER_API(Reader*)CreateFormatReader(const char *filename) {
    return Registry::CreateReader(filename);
}