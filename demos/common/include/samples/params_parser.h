#pragma once
#include <vector>
#include <map>

struct Param {
    Param() {
        isPresent = false;
    }

    Param(std::string name,
           std::string shortName,
           std::string defVal,
           std::string description,
           bool isRequired,
           int expectedArgumentsNumber) {
        this->name = name;
        this->shortName = shortName;
        values.push_back(defVal);
        this->isPresent = false;
        this->description = description;
        this->isRequired = isRequired;
        this->expectedArgumentsNumber = expectedArgumentsNumber;
    }

    bool IsNothing() {}

    std::string name;
    std::string shortName;
    std::vector<std::string> values;
    std::string description;
    bool isPresent;
    bool isRequired;
    int expectedArgumentsNumber;

    static Param Nothing;
};

class ParamsParser {
public:
    std::string parse(int argc, char* const argv[], bool commaSeparated = false);
    void addParam(std::string name,
                  std::string shortName,
                  std::string defVal,
                  std::string description,
                  bool isRequired,
                  int expectedArgumentsNumber = -1);

    bool isPresent(const char* name);
    std::vector<std::string> getStrings(const char * name);
    std::string getString(const char * name);
    std::string operator[](const char* name) { return getString(name); }
    int getInt(const char * name);
    double getDouble(const char * name);
    bool getBool(const char * name);

    std::string GetParamsHelp();

protected:
    std::map<std::string, Param> params;
    Param& getParam(const char* name);
};
