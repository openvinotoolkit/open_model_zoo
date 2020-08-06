// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality
 * @file args_helper.hpp
 */

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#define DEFINE_INPUT_FLAGS \
static const char input_message[] = "Required. An input to process. The input must be a single image, a folder of " \
    "images or anything that cv::VideoCapture can process."; \
static const char loop_message[] = "Optional. Enable reading the input in a loop."; \
DEFINE_string(i, "", input_message); \
DEFINE_bool(loop, false, loop_message);

/**
* @brief This function checks input args and existence of specified files in a given folder
* @param arg path to a file to be checked for existence
* @return files updated vector of verified input files
*/
void readInputFilesArguments(std::vector<std::string>& files, const std::string& arg);

/**
* @brief This function finds -i/--i key in input args
*        It's necessary to process multiple values for single key
* @return files updated vector of verified input files
*/
void parseInputFilesArguments(std::vector<std::string>& files);

std::vector<std::string> split(const std::string& s, char delim);

std::vector<std::string> parseDevices(const std::string& device_string);

std::map<std::string, uint32_t> parseValuePerDevice(const std::set<std::string>& devices,
                                                    const std::string& values_string);
