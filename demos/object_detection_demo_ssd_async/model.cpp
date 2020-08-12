#include "model.hpp"

Model::Model(const InferenceEngine::Core &ie, std::string networkModel) {
    this->cnnNetwork = ie.ReadNetwork(networkModel);
}

InferenceEngine::CNNNetwork Model::getCnnNetwork()const {
    return this->cnnNetwork;
}

void Model::loadLables(std::string labelsFile) {
    if (!labelsFile.empty()) {
        std::ifstream inputFile(labelsFile);
        std::string label;
        while (std::getline(inputFile, label)) {
            (this->labels).push_back(label);
        }
        if ((this->labels).empty())
            throw std::logic_error("File empty or not found: " + labelsFile);
    }
}

std::size_t Model::getInputHeight()const {
    return this->inputHeight;
}

std::size_t Model::getInputWidth()const {
    return this->inputWidth;
}

const std::map<std::string, InferenceEngine::SizeVector>& Model::getInputs()const {
    return this->inputs;
}
const std::map< std::string, InferenceEngine::DataPtr&>& Model::getOutputs()const {
    return this->outputs;
}

const std::vector< std::string>& Model::getOutputsNames()const {
    return this->outputsNames;
}

 std::string Model::getImageInputName()const {
    return this->imageInputName;
}
