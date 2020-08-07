#include "models.hpp"

Model::Model(const InferenceEngine::Core &ie, std::string FLAGS_m) {

    this->cnnNetwork = ie.ReadNetwork(FLAGS_m);
}

InferenceEngine::CNNNetwork Model::getCnnNetwork() {

    return this->cnnNetwork;

}

std::size_t Model::getInputHeight() {
    return this->inputHeight;
}

std::size_t Model::getInputWidth() {

    return this->inputWidth;

}

std::map<std::string, InferenceEngine::SizeVector >& Model::getInputs(){

    return this->inputs;
}
std::map<const std::string, InferenceEngine::DataPtr&>& Model::getOutputs() {
    return this->outputs;
}

std::vector<const std::string>& Model::getOutputsNames() {//больше не знаю способа, возврата вектора имен выходов


    for (auto it = (this->outputs).begin(); it != (this->outputs).end(); it++)
    {
        (this->outputsNames).push_back(it->first);
    }

    return this->outputsNames;

}

std::string Ssd::getImageInfoInputName() {

    return this->imageInfoInputName;

}
