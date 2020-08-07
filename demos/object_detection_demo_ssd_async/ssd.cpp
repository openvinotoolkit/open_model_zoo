#include "ssd.hpp"
#include "models.hpp"

Ssd::Ssd(const InferenceEngine::Core &ie, std::string FLAGS_m):Model(ie, FLAGS_m) {
    /** Set batch size to 1 **/
    slog::info << "Batch size is forced to 1." << slog::endl;
    (this->cnnNetwork).setBatchSize(1);
    this->maxProposalCount=0;
    this->objectSize=0;
}

void  Ssd::prepareInputBlobs() {
    InferenceEngine::InputsDataMap inputInfo((this->cnnNetwork).getInputsInfo());

    for (const auto& inputInfoItem : inputInfo) {
        if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // 1st input contains images

            inputInfoItem.second->setPrecision(InferenceEngine::Precision::U8);
            if (FLAGS_auto_resize) {
                inputInfoItem.second->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
                inputInfoItem.second->getInputData()->setLayout(InferenceEngine::Layout::NHWC);
            }
            else {
                inputInfoItem.second->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
            }

            inputs.insert(std::pair<std::string, InferenceEngine::SizeVector >(inputInfoItem.first, inputInfoItem.second->getTensorDesc().getDims()));
            const InferenceEngine::TensorDesc& inputDesc = inputInfoItem.second->getTensorDesc();
            this->inputHeight = getTensorHeight(inputDesc);
            this->inputWidth = getTensorWidth(inputDesc);
        }
        else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // 2nd input contains image info
            this->imageInfoInputName = inputInfoItem.first;
            inputInfoItem.second->setPrecision(InferenceEngine::Precision::FP32);
        }
        else {
            throw std::logic_error("Unsupported " +
                std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
                "input layer '" + inputInfoItem.first + "'. "
                "Only 2D and 4D input layers are supported");
        }
    }



}

void Ssd::prepareOutputBlobs() {
    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    InferenceEngine::OutputsDataMap outputInfo((this->cnnNetwork).getOutputsInfo());
    if (outputInfo.size() != 1) {
        throw std::logic_error("This demo accepts networks having only one output");
    }
    (this->outputs).insert(std::pair<std::string, InferenceEngine::DataPtr&>(outputInfo.begin()->first, outputInfo.begin()->second));

    int num_classes = 0;

    if (auto ngraphFunction = (this->cnnNetwork).getFunction()) {
        for (const auto op : ngraphFunction->get_ops()) {
            if (op->get_friendly_name() == (this->outputs).begin()->first) {
                auto detOutput = std::dynamic_pointer_cast<ngraph::op::DetectionOutput>(op);
                if (!detOutput) {
                    THROW_IE_EXCEPTION << "Object Detection network output layer(" + op->get_friendly_name() +
                        ") should be DetectionOutput, but was " + op->get_type_info().name;
                }

                num_classes = detOutput->get_attrs().num_classes;
                break;
            }
        }
    }
    /* else if (!labels.empty()) {
         throw std::logic_error("Class labels are not supported with IR version older than 10");
     }

     if (!labels.empty() && static_cast<int>(labels.size()) != num_classes) {
         if (static_cast<int>(labels.size()) == (num_classes - 1))  // if network assumes default "background" class, having no label
             labels.insert(labels.begin(), "fake");
         else {
             throw std::logic_error("The number of labels is different from numbers of model classes");
         }
     }*/
    InferenceEngine::SizeVector outputDims = (this->outputs).begin()->second->getTensorDesc().getDims();
    this->maxProposalCount = outputDims[2];// поля класса ssd
    this->objectSize = outputDims[3];
    if (this->objectSize != 7) {
        throw std::logic_error("Output should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD");
    }
    (this->outputs).begin()->second->setPrecision(InferenceEngine::Precision::FP32);
    (this->outputs).begin()->second->setLayout(InferenceEngine::Layout::NCHW);

}

void Ssd::setConstInput(InferenceEngine::InferRequest::Ptr& inferReq) {
    auto blob = inferReq->GetBlob(this->imageInfoInputName);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->wmap();
    auto data = blobMapped.as<float*>();
    data[0] = static_cast<float>(inputHeight);  // height
    data[1] = static_cast<float>(inputWidth);  // width
    data[2] = 1;

}

void Ssd::processOutput(std::map<const std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, const size_t width, const size_t height, std::vector<std::string>& labels, bool FLAGS_r, double threshold) {
    InferenceEngine::LockedMemory<const void> outputMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>((outputs.begin()->second))->rmap();
    const float* detections = outputMapped.as<float*>();
    for (int i = 0; i < this->maxProposalCount; i++) {
        float image_id = detections[i * this->objectSize + 0];
        if (image_id < 0) {
            break;
        }

        float confidence = detections[i * this->objectSize + 2];
        auto label = static_cast<int>(detections[i * this->objectSize + 1]);
        float xmin = detections[i * this->objectSize + 3] * width;
        float ymin = detections[i * this->objectSize + 4] * height;
        float xmax = detections[i * this->objectSize + 5] * width;
        float ymax = detections[i * this->objectSize + 6] * height;

        if (FLAGS_r) {
            std::cout << "[" << i << "," << label << "] element, prob = " << confidence <<
                "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
                << ((confidence > threshold) ? " WILL BE RENDERED!" : "") << std::endl;
        }

        if (confidence > threshold) {
            /** Drawing only objects when > confidence_threshold probability **/
            std::ostringstream conf;
            conf << ":" << std::fixed << std::setprecision(3) << confidence;
            cv::putText(frame,
                (!labels.empty() ? labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                cv::Point2f(xmin, ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                cv::Scalar(0, 0, 255));
            cv::rectangle(frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax),
                cv::Scalar(0, 0, 255));
        }
    }


}

const int Ssd::getMaxProposalCount() {

    return this->maxProposalCount;
}

const int Ssd::getObjectSize() {

    return this->objectSize;

}


