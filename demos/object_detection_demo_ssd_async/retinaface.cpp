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

   

    if (!(this->labels).empty() && static_cast<int>((this->labels).size()) != ((params.begin())->second)->classes) {
        throw std::runtime_error("The number of labels is different from numbers of model classes");
    }

}

void Retinaface::setConstInput(InferenceEngine::InferRequest::Ptr& inferReq) {}

void Retinaface::processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, bool printOutput, double threshold) {

    // Processing results of the CURRENT request
    std::vector<DetectionObject> objects;
    // Parsing outputs
    for (auto& output : outputs) {
        this->parseYOLOV3Output(output.first, output.second, this->inputHeight, this->inputWidth, frame.size().height, frame.size().width, threshold, objects);
    }
    // Filtering overlapping boxes
    double FLAGS_iou_t = 0;
    std::sort(objects.begin(), objects.end(), std::greater<DetectionObject>());
    for (size_t i = 0; i < objects.size(); ++i) {
        if (objects[i].confidence == 0)
            continue;
        for (size_t j = i + 1; j < objects.size(); ++j)
            if (IntersectionOverUnion(objects[i], objects[j]) >= FLAGS_iou_t)
                objects[j].confidence = 0;
    }
    // Drawing boxes
    for (auto& object : objects) {
        if (object.confidence < threshold)
            continue;
        auto label = object.class_id;
        float confidence = object.confidence;
        if (printOutput) {
            std::cout << "[" << label << "] element, prob = " << confidence <<
                "    (" << object.xmin << "," << object.ymin << ")-(" << object.xmax << "," << object.ymax << ")"
                << ((confidence > threshold) ? " WILL BE RENDERED!" : "") << std::endl;
        }
        if (confidence > threshold) {
            /** Drawing only objects when >confidence_threshold probability **/
            std::ostringstream conf;
            conf << ":" << std::fixed << std::setprecision(3) << confidence;
            cv::putText(frame,
                (!(this->labels).empty() ? this->labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin - 5)), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                cv::Scalar(0, 0, 255));
            cv::rectangle(frame, cv::Point2f(static_cast<float>(object.xmin), static_cast<float>(object.ymin)),
                cv::Point2f(static_cast<float>(object.xmax), static_cast<float>(object.ymax)), cv::Scalar(0, 0, 255));
        }
    }


}
