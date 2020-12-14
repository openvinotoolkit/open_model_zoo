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
           int expectedArgumentsNumber,
           bool unsplittable) {
        this->name = name;
        this->shortName = shortName;
        values.push_back(defVal);
        this->isPresent = false;
        this->description = description;
        this->isRequired = isRequired;
        this->expectedArgumentsNumber = expectedArgumentsNumber;
        this->unsplittable = unsplittable;
    }

    bool IsNothing() {}

    std::string name;
    std::string shortName;
    std::vector<std::string> values;
    std::string description;
    bool isPresent;
    bool isRequired;
    int expectedArgumentsNumber;
    bool unsplittable;

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
                  int expectedArgumentsNumber = -1,
                  bool unsplittable = false);

    /// Returns true if this parameter was explicitely listed in command line (even without arguments)
    /// Note that it differs from getBool function - this operator only checks for keyword presence in command line
    /// while getBool returns actual boolean value provided as argument to command line option
    bool operator[](const char* name);
    std::vector<std::string> getStringsArray(const char * name);
    std::string getStr(const char * name);
    int getInt(const char * name);
    double getDbl(const char * name);
    bool getBool(const char * name);

    std::string GetParamsHelp();

protected:
    std::map<std::string, Param> params;
    Param& getParam(const char* name);
};
