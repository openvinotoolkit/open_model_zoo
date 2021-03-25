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

#include <opencv2/core/types.hpp>

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

cv::Size stringToSize(const std::string& str);
