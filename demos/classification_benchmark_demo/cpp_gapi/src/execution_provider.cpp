// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/args_helper.hpp"
#include "execution_providers.hpp"


execution_providers_t createProvidersFromString(const std::string &str){
#ifndef GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
    if (!str.empty()) {
        std::cerr << "G-API execution providers are not available in OPENCV: " << CV_VERSION
                  << "\nPlease make sure you are using 4.9.0 at least" << std::endl;
        exit(-1);
    }
    return {};
#else // GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
    execution_providers_t providers;
    std::stringstream params_list(str);
    try {
        std::string line;
        while (std::getline(params_list, line, ';')) {
            std::vector<std::string> splitted_line = split(line, ':');
            if (splitted_line.size() != 2) {
                throw std::runtime_error("Cannot parse execution provider from string: " + line +
                                        ". Provider name and its device must be separated by \":\" - <provider>:<device>");
            }

            providers.emplace(std::move(splitted_line[0]), std::move(splitted_line[1]));
        }
    } catch (const std::exception& ex) {
        std::cerr << "Invalid -ep format: " << ex.what() << std::endl;
        exit(-1);
    }
    return providers;
#endif // GAPI_IE_EXECUTION_PROVIDERS_AVAILABLE
}
