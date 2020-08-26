#include "retinaface.hpp"
#include "model.hpp"
#include <ngraph/ngraph.hpp>
#include <iostream>
#include <map>
#include <samples/slog.hpp>


Retinaface::Retinaface(const InferenceEngine::Core& ie, std::string networkModel) :Model(ie, networkModel) {
    /** Set batch size to 1 **/
    slog::info << "Batch size is forced to 1." << slog::endl;
    (this->cnnNetwork).setBatchSize(1);

}

void Retinaface::prepareInputBlobs(bool autoResize) {
    slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
    InferenceEngine::InputsDataMap inputInfo(this->cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks that have only one input");
    }
    InferenceEngine::InputInfo::Ptr& input = inputInfo.begin()->second;
    this->imageInputName = inputInfo.begin()->first;
    input->setPrecision(InferenceEngine::Precision::U8);
    if (autoResize) {
        input->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
        input->getInputData()->setLayout(InferenceEngine::Layout::NHWC);
    }
    else {
        input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
    }
    this->inputs.insert(std::pair<std::string, InferenceEngine::SizeVector >(inputInfo.begin()->first, inputInfo.begin()->second->getTensorDesc().getDims()));
    const InferenceEngine::TensorDesc& inputDesc = inputInfo.begin()->second->getTensorDesc();
    this->inputHeight = getTensorHeight(inputDesc);
    this->inputWidth = getTensorWidth(inputDesc);
}

void Retinaface::prepareOutputBlobs() {

    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::NCHW);
        (this->outputs).insert(std::pair<std::string, InferenceEngine::DataPtr&>(output.first, output.second));
        (this->outputsNames).push_back(output.first);
    }

    if (this->outputs.size()!=9 || this->outputs.size() != 12)
          throw std::logic_error("Expected 12 or 9 output blobs");
   

   /* if (!(this->labels).empty() && static_cast<int>((this->labels).size()) != ((params.begin())->second)->classes) {
        throw std::runtime_error("The number of labels is different from numbers of model classes");
    }*/

}

void Retinaface::setConstInput(InferenceEngine::InferRequest::Ptr& inferReq) {}

void Retinaface::processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, bool printOutput, double threshold) {

  

}
