/*
// Copyright (C) 2023-2024 Intel Corporation
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

#include <fstream>
#include <map>

#include "utils/common.hpp"

std::vector<unsigned> loadClassIndices(const std::string &groundtruth_filepath,
                                       const std::vector<std::string> &imageNames)
{
    std::vector<unsigned> classIndices;
    if (groundtruth_filepath.empty()) {
        classIndices.resize(imageNames.size(), 0);
    } else {
        std::map<std::string, unsigned> classIndicesMap;
        std::ifstream inputGtFile(groundtruth_filepath);
        if (!inputGtFile.is_open()) {
            throw std::runtime_error("Can't open the ground truth file: " + groundtruth_filepath);
        }

        std::string line;
        while (std::getline(inputGtFile, line)) {
            size_t separatorIdx = line.find(' ');
            if (separatorIdx == std::string::npos) {
                throw std::runtime_error("The ground truth file has incorrect format.");
            }
            std::string imagePath = line.substr(0, separatorIdx);
            size_t imagePathEndIdx = imagePath.rfind('/');
            unsigned classIndex = static_cast<unsigned>(std::stoul(line.substr(separatorIdx + 1)));
            if ((imagePathEndIdx != 1 || imagePath[0] != '.') && imagePathEndIdx != std::string::npos) {
                throw std::runtime_error("The ground truth file has incorrect format.");
            }
            // std::map type for classIndicesMap guarantees to sort out images by name.
            // The same logic is applied in openImagesCapture() for DirReader source type,
            // which produces data for sorted pictures.
            // To be coherent in detection of ground truth for pictures we have to
            // use the same sorting approach for a source and ground truth data
            // If you're going to copy paste this code, remember that pictures need to be sorted
            classIndicesMap.insert({imagePath.substr(imagePathEndIdx + 1), classIndex});
        }

        for (size_t i = 0; i < imageNames.size(); i++) {
            auto imageSearchResult = classIndicesMap.find(imageNames[i]);
            if (imageSearchResult != classIndicesMap.end()) {
                classIndices.push_back(imageSearchResult->second);
            } else {
                throw std::runtime_error("No class specified for image " + imageNames[i]);
            }
        }
    }
    return classIndices;
}
