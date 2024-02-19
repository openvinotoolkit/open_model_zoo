// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <fstream>

#include <opencv2/opencv.hpp>

static std::string getType(const std::string& header) {
    std::string field = "'descr':";
    int idx = header.find(field);
    CV_Assert(idx != -1);

    int from = header.find('\'', idx + field.size()) + 1;
    int to = header.find('\'', from);
    return header.substr(from, to - from);
}

static std::string getFortranOrder(const std::string& header) {
    std::string field = "'fortran_order':";
    int idx = header.find(field);
    CV_Assert(idx != -1);

    int from = header.find_last_of(' ', idx + field.size()) + 1;
    int to = header.find(',', from);
    return header.substr(from, to - from);
}

static std::vector<int> getShape(const std::string& header) {
    std::string field = "'shape':";
    int idx = header.find(field);
    CV_Assert(idx != -1);

    int from = header.find('(', idx + field.size()) + 1;
    int to = header.find(')', from);

    std::string shapeStr = header.substr(from, to - from);
    if (shapeStr.empty())
        return std::vector<int>(1, 1);

    // Remove all commas.
    shapeStr.erase(std::remove(shapeStr.begin(), shapeStr.end(), ','),
                   shapeStr.end());

    std::istringstream ss(shapeStr);
    int value;

    std::vector<int> shape;
    while (ss >> value) {
        shape.push_back(value);
    }
    return shape;
}

cv::Mat blobFromNPY(const std::string& path) {
    std::ifstream ifs(path.c_str(), std::ios::binary);
    CV_Assert(ifs.is_open());

    std::string magic(6, '*');
    ifs.read(&magic[0], magic.size());
    CV_Assert(magic == "\x93NUMPY");

    ifs.ignore(1);  // Skip major version byte.
    ifs.ignore(1);  // Skip minor version byte.

    unsigned short headerSize;
    ifs.read((char*)&headerSize, sizeof(headerSize));

    std::string header(headerSize, '*');
    ifs.read(&header[0], header.size());

    // Extract data type.
    CV_Assert(getFortranOrder(header) == "False");
    std::vector<int> shape = getShape(header);

    std::string dataType = getType(header);
    cv::Mat blob;
    if (dataType == "<f4")
        blob.create(shape, CV_32F);
    else if (dataType == "|b1")
        blob.create(shape, CV_8U);
    else if (dataType == "<f8")
        blob.create(shape, CV_64F);
    else
        throw std::logic_error("Unexpected numpy data type: " + dataType);

    ifs.read((char*)blob.data, blob.total() * blob.elemSize());
    CV_Assert((size_t)ifs.gcount() == blob.total() * blob.elemSize());

    return blob;
}
