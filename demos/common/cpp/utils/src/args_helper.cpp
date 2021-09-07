// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/args_helper.hpp"
#include "utils/slog.hpp"

#ifdef _WIN32
#include "w_dirent.hpp"
#else
#include <dirent.h>
#endif

#include <gflags/gflags.h>

#include <sys/stat.h>
#include <map>

#include <algorithm>
#include <cctype>
#include <sstream>

void readInputFilesArguments(std::vector<std::string>& files, const std::string& arg) {
    struct stat sb;
    if (stat(arg.c_str(), &sb) != 0) {
        if (arg.compare(0, 5, "rtsp:") != 0) {
            slog::warn << "File " << arg << " cannot be opened!" << slog::endl;
            return;
        }
    }
    if (S_ISDIR(sb.st_mode)) {
        DIR *dp;
        dp = opendir(arg.c_str());
        if (dp == nullptr) {
            slog::warn << "Directory " << arg << " cannot be opened!" << slog::endl;
            return;
        }

        struct dirent *ep;
        while (nullptr != (ep = readdir(dp))) {
            std::string fileName = ep->d_name;
            if (fileName == "." || fileName == "..") continue;
            files.push_back(arg + "/" + ep->d_name);
        }
        closedir(dp);
    } else {
        files.push_back(arg);
    }

    if (files.size() < 20) {
        slog::info << "Files were added: " << files.size() << slog::endl;
        for (std::string filePath : files) {
            slog::info << "    " << filePath << slog::endl;
        }
    } else {
        slog::info << "Files were added: " << files.size() << ". Too many to display each of them." << slog::endl;
    }
}

void parseInputFilesArguments(std::vector<std::string>& files) {
    std::vector<std::string> args = gflags::GetArgvs();
    bool readArguments = false;
    for (size_t i = 0; i < args.size(); i++) {
        if (args.at(i) == "-i" || args.at(i) == "--i") {
            readArguments = true;
            continue;
        }
        if (!readArguments) {
            continue;
        }
        if (args.at(i).c_str()[0] == '-') {
            break;
        }
        readInputFilesArguments(files, args.at(i));
    }
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::vector<std::string> parseDevices(const std::string& device_string) {
    const std::string::size_type colon_position = device_string.find(":");
    if (colon_position != std::string::npos) {
        std::string device_type = device_string.substr(0, colon_position);
        if (device_type == "HETERO" || device_type == "MULTI") {
            std::string comma_separated_devices = device_string.substr(colon_position + 1);
            std::vector<std::string> devices = split(comma_separated_devices, ',');
            for (auto& device : devices)
                device = device.substr(0, device.find("("));
            return devices;
        }
    }
    return {device_string};
}

// Format: <device1>:<value1>,<device2>:<value2> or just <value>
std::map<std::string, uint32_t> parseValuePerDevice(const std::set<std::string>& devices,
                                                    const std::string& values_string) {
    auto values_string_upper = values_string;
    std::transform(values_string_upper.begin(),
                   values_string_upper.end(),
                   values_string_upper.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    std::map<std::string, uint32_t> result;
    auto device_value_strings = split(values_string_upper, ',');
    for (auto& device_value_string : device_value_strings) {
        auto device_value_vec =  split(device_value_string, ':');
        if (device_value_vec.size() == 2) {
            auto it = std::find(devices.begin(), devices.end(), device_value_vec.at(0));
            if (it != devices.end()) {
                result[device_value_vec.at(0)] = std::stoi(device_value_vec.at(1));
            }
        } else if (device_value_vec.size() == 1) {
            uint32_t value = std::stoi(device_value_vec.at(0));
            for (const auto& device : devices) {
                result[device] = value;
            }
        } else if (device_value_vec.size() != 0) {
            throw std::runtime_error("Unknown string format: " + values_string);
        }
    }
    return result;
}

cv::Size stringToSize(const std::string& str) {
    std::vector<std::string> strings = split(str, 'x');
    if (strings.size() != 2) {
        throw std::invalid_argument("Can't convert std::string to cv::Size. The string must contain exactly one x");
    }
    return {std::stoi(strings[0]), std::stoi(strings[1])};
}
