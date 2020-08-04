#pragma once
#include "Models.h"

class ssd_async :public model
{
private:
    std::string imageInfoInputName;
    const int maxProposalCount;
    const int objectSize;

public:
    
    ssd_async(const Core& ie, std::string FLAGS_m) :model(ie, FLAGS_m) {}

    

	void PreparingInputBlobs() {
		InputsDataMap inputInfo(cnnNetwork.getInputsInfo());

      

        for (const auto& inputInfoItem : inputInfo) {
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // 1st input contains images
               
                inputInfoItem.second->setPrecision(Precision::U8);
                if (FLAGS_auto_resize) {
                    inputInfoItem.second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
                    inputInfoItem.second->getInputData()->setLayout(Layout::NHWC);
                }
                else {
                    inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
                }
               
                inputs.insert(pair<std::string, InferenceEngine::SizeVector >(inputInfoItem.first, inputInfoItem.second->getTensorDesc().getDims()));// если итератор заходит в это условие больше 1 раза, то вставить условие на пустоту inputs, и изменить заполнение
            }
            else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // 2nd input contains image info
                imageInfoInputName = inputInfoItem.first;
                inputInfoItem.second->setPrecision(Precision::FP32);
            }
            else {
                throw std::logic_error("Unsupported " +
                    std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
                    "input layer '" + inputInfoItem.first + "'. "
                    "Only 2D and 4D input layers are supported");
            }
        }



	}



    void PreparingOutputBlobs(std::vector<std::string>& labels) {
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("This demo accepts networks having only one output");
        }
        DataPtr& output = outputInfo.begin()->second;
        outputs = outputInfo.begin()->first;

        int num_classes = 0;

        if (auto ngraphFunction = cnnNetwork.getFunction()) {
            for (const auto op : ngraphFunction->get_ops()) {
                if (op->get_friendly_name() == outputs[0]) {
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
        else if (!labels.empty()) {
            throw std::logic_error("Class labels are not supported with IR version older than 10");
        }

        if (!labels.empty() && static_cast<int>(labels.size()) != num_classes) {
            if (static_cast<int>(labels.size()) == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else {
                throw std::logic_error("The number of labels is different from numbers of model classes");
            }
        }
        const SizeVector outputDims = output->getTensorDesc().getDims();
         maxProposalCount = outputDims[2];// поля класса ssd
         objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        output->setPrecision(Precision::FP32);
        output->setLayout(Layout::NCHW);
    
    }


    InferRequest::Ptr CreateInferRequest(ExecutableNetwork userSpecifiedExecNetwork, ExecutableNetwork minLatencyExecNetwork, uint32 FLAGS_nireq) {
    

        std::vector<InferRequest::Ptr> userSpecifiedInferRequests;
        for (unsigned infReqId = 0; infReqId < FLAGS_nireq; ++infReqId) {
            userSpecifiedInferRequests.push_back(userSpecifiedExecNetwork.CreateInferRequestPtr());
        }

        InferRequest::Ptr minLatencyInferRequest = minLatencyExecNetwork.CreateInferRequestPtr();

        /* it's enough just to set image info input (if used in the model) only once */
        if (!imageInfoInputName.empty()) {
            auto setImgInfoBlob = [&](const InferRequest::Ptr& inferReq) {
                auto blob = inferReq->GetBlob(imageInfoInputName);
                LockedMemory<void> blobMapped = as<MemoryBlob>(blob)->wmap();
                auto data = blobMapped.as<float*>();
                data[0] = static_cast<float>(inputs.begin()->second[0]);  // height
                data[1] = static_cast<float>(inputs.begin()->second[1]);  // width может наоборот w h 
                data[2] = 1;
            };

            for (const InferRequest::Ptr& requestPtr : userSpecifiedInferRequests) {
                setImgInfoBlob(requestPtr);
            }
            setImgInfoBlob(minLatencyInferRequest);
        }
    
        return minLatencyInferRequest;
    }


    void ProcessingOutputBlobs(const float* detections, const size_t width, const size_t height, std::vector<std::string>& labels,bool FLAGS_r,double FLAGS_t) {
    
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];
            if (image_id < 0) {
                break;
            }

            float confidence = detections[i * objectSize + 2];
            auto label = static_cast<int>(detections[i * objectSize + 1]);
            float xmin = detections[i * objectSize + 3] * width;
            float ymin = detections[i * objectSize + 4] * height;
            float xmax = detections[i * objectSize + 5] * width;
            float ymax = detections[i * objectSize + 6] * height;

            if (FLAGS_r) {
                std::cout << "[" << i << "," << label << "] element, prob = " << confidence <<
                    "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
                    << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
            }

            if (confidence > FLAGS_t) {
                /** Drawing only objects when > confidence_threshold probability **/
                std::ostringstream conf;
                conf << ":" << std::fixed << std::setprecision(3) << confidence;
                cv::putText(requestResult.frame,
                    (!labels.empty() ? labels[label] : std::string("label #") + std::to_string(label)) + conf.str(),
                    cv::Point2f(xmin, ymin - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1,
                    cv::Scalar(0, 0, 255));
                cv::rectangle(requestResult.frame, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax),
                    cv::Scalar(0, 0, 255));
            }
        }


    }
	
    const int get_maxProposalCount() {

        return this->maxProposalCount;
    }

    const int get_objectSize() {

        return this->objectSize;

    }

    std::string get_imageInputName() {

        return this->(inputs.begin()->first);

    }
    const std::string get_outputName() {

        return (this->outputs).first;
    }

    //std::size_t get_height();
};