#include "samples/params_parser.h"
#include <string>
#include <sstream>
#include <iomanip>

Param Param::Nothing;

std::string ParamsParser::parse(int argc, char* const argv[], bool commaSeparated) {
    Param* currentParam = nullptr;
    bool isFirstVal = false;

    for (int i = 1; i < argc; i++) {
        //--- Searching for the key
        if (!strncmp(argv[i], "--", 2)) {
            currentParam = NULL;
            auto it = params.find(argv[i] + 2);
            if (it != params.end()) {
                currentParam = &(it->second);
                currentParam->isPresent = true;
                currentParam->values.clear();
                isFirstVal = true;
            }
            else {
                return std::string("Invalid command line argument found: ") + std::string(argv[i]);
            }
        }
        else if (!strncmp(argv[i], "-", 1)) {
            currentParam = NULL;
            auto it = params.begin();
            for (; it != params.end(); it++) {
                if (it->second.shortName == argv[i] + 1) {
                    currentParam = &(it->second);
                    currentParam->isPresent = true;
                    currentParam->values.clear();
                    isFirstVal = true;
                    break;
                }
            }
            if (it == params.end()) {
                return std::string("Invalid command line argument found: ") + std::string(argv[i]);
            }

        }
        else {
            //--- Reading values
            if (currentParam) {
                if (commaSeparated) {
                    // Comma-separated multiple parameters
                    if (!isFirstVal) {
                        throw std::invalid_argument(std::string("Invalid command line argument found: ") + std::string(argv[i]));
                    }
                    else {
                        std::string blob = argv[i];
                        currentParam->values.clear();
                        if (currentParam->unsplittable) {
                            currentParam->values.push_back(blob);
                        }
                        else {
                            size_t st = 0;
                            size_t end = 0;
                            while ((end = blob.find(',', st)) != std::string::npos) {
                                currentParam->values.push_back(blob.substr(st, end - st));
                                st = end + 1;
                            }
                            currentParam->values.push_back(blob.substr(st));
                        }
                    }
                }
                else {
                    // Space-separated multiple parameters
                    if (isFirstVal) {
                        isFirstVal = false;
                        currentParam->values.clear();
                    }
                    currentParam->values.push_back(argv[i]);
                }
            }
            else {
                return std::string("Invalid command line argument found: ") + std::string(argv[i]);
            }
        }
    }

    std::string paramsMissing = "";
    std::string errorMessage = "";
    for (auto& paramsPair : params) {
        if (paramsPair.second.isPresent) {
            if (paramsPair.second.expectedArgumentsNumber != -1 &&
                paramsPair.second.values.size() != paramsPair.second.expectedArgumentsNumber) {
                errorMessage += "Invalid number of arguments for parameter " + std::string(paramsPair.second.name) +
                    ". Expected: " + std::to_string(paramsPair.second.expectedArgumentsNumber) +
                    ", got: " + std::to_string(paramsPair.second.values.size()) + ".\n";
            }
        }
        else {
            if (paramsPair.second.isRequired) {
                paramsMissing += paramsPair.second.name + ", ";
            }
        }
    }
    if (paramsMissing != "" || errorMessage != "") {
        return errorMessage + std::string("Some mandatory command line parameters are missing: ") +
            paramsMissing.substr(0, paramsMissing.length() - 2);
    }
    return "";
}

void ParamsParser::addParam(std::string name, std::string shortName, std::string defVal,
                            std::string description, bool isRequired, int expectedArgumentsNumber, bool unsplittable) {
    params.emplace(name, Param(name, shortName, defVal, description, isRequired, expectedArgumentsNumber, unsplittable));
}


Param& ParamsParser::getParam(const char* name) {
    auto it = params.find(name);
    if (it == params.end()) {
        throw std::invalid_argument(std::string("Invalid parameter requested: ") + std::string(name));
    }
    return it->second;
}

std::vector<std::string> ParamsParser::getStringsArray(const char* name) {
    return getParam(name).values;
}

std::string ParamsParser::getStr(const char* name) {
    auto& param = getParam(name);
    if (param.values.size() == 0) {
        throw std::out_of_range(std::string("No value for parameter: ") + std::string(name));
    }
    return param.values[0];
}


bool ParamsParser::operator[](const char* name) {
    return getParam(name).isPresent;
}

int ParamsParser::getInt(const char* name) {
    return std::stoi(getStr(name));
}

double ParamsParser::getDbl(const char* name) {
    return std::stof(getStr(name));
}

bool ParamsParser::getBool(const char* name) {
    auto val = getStr(name);
    for (auto & c : val)
        c = toupper(c);
    return val == "TRUE" || std::stoi(val) != 0;
}

std::string ParamsParser::GetParamsHelp() {
    std::stringstream retVal;
    for (auto& paramsPair : params) {
        retVal << "--";
        retVal << std::setw(20) << std::left << (paramsPair.second.name +
            (paramsPair.second.shortName != "" ? std::string(",-") + paramsPair.second.shortName : ""));
        retVal << " " << paramsPair.second.description << "\n";
    }
    return retVal.str();
}
