#include "yolov3.hpp"
#include "model.hpp"
#include <ngraph/ngraph.hpp>
#include <iostream>
#include <map>
#include <samples/slog.hpp>


double IntersectionOverUnion(const DetectionObject& box_1, const DetectionObject& box_2) {
    double width_of_overlap_area = fmin(box_1.xmax, box_2.xmax) - fmax(box_1.xmin, box_2.xmin);
    double height_of_overlap_area = fmin(box_1.ymax, box_2.ymax) - fmax(box_1.ymin, box_2.ymin);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;
    double box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin);
    double box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin);
    double area_of_union = box_1_area + box_2_area - area_of_overlap;
    return area_of_overlap / area_of_union;
}

 static int EntryIndex(int side, int lcoords, int lclasses, int location, int entry) {
    int n = location / (side * side);
    int loc = location % (side * side);
    return n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc;
}

Yolov3::Yolov3(const InferenceEngine::Core& ie, std::string networkModel):Model(ie, networkModel) {
    /** Set batch size to 1 **/
    slog::info << "Batch size is forced to 1." << slog::endl;
    (this->cnnNetwork).setBatchSize(1);

}

void Yolov3::parseYOLOV3Output( const std::string& output_name,
    const InferenceEngine::Blob::Ptr& blob, const unsigned long resized_im_h,
    const unsigned long resized_im_w, const unsigned long original_im_h,
    const unsigned long original_im_w,
    const double threshold, std::vector<DetectionObject>& objects) {

    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    if (out_blob_h != out_blob_w)
        throw std::runtime_error("Invalid size of output " + output_name +
            " It should be in NCHW layout and H should be equal to W. Current H = " + std::to_string(out_blob_h) +
            ", current W = " + std::to_string(out_blob_h));

    // --------------------------- Extracting layer parameters -------------------------------------
    Params params;
    if (auto ngraphFunction = this->cnnNetwork.getFunction()) {
        for (const auto op : ngraphFunction->get_ops()) {
            if (op->get_friendly_name() == output_name) {
                auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                if (!regionYolo) {
                    throw std::runtime_error("Invalid output type: " +
                        std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                }

                params = regionYolo;
                break;
            }
        }
    }
    else {
        throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
    }

    auto side = out_blob_h;
    auto side_square = side * side;
    const float* output_blob = blob->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    // --------------------------- Parsing YOLO Region output -------------------------------------
    for (int i = 0; i < side_square; ++i) {
        int row = i / side;
        int col = i % side;
        for (int n = 0; n < params.num; ++n) {
            int obj_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, params.coords);
            int box_index = EntryIndex(side, params.coords, params.classes, n * side * side + i, 0);
            float scale = output_blob[obj_index];
            if (scale < threshold)
                continue;
            double x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w;
            double y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h;
            double height = std::exp(output_blob[box_index + 3 * side_square]) * params.anchors[2 * n + 1];
            double width = std::exp(output_blob[box_index + 2 * side_square]) * params.anchors[2 * n];
            for (int j = 0; j < params.classes; ++j) {
                int class_index = EntryIndex(side, params.coords, params.classes, n * side_square + i, params.coords + 1 + j);
                float prob = scale * output_blob[class_index];
                if (prob < threshold)
                    continue;
                DetectionObject obj(x, y, height, width, j, prob,
                    static_cast<float>(original_im_h) / static_cast<float>(resized_im_h),
                    static_cast<float>(original_im_w) / static_cast<float>(resized_im_w));
                objects.push_back(obj);
            }
        }
    }
}

void Yolov3::prepareInputBlobs(bool autoResize) {
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

void Yolov3::prepareOutputBlobs() {

    slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    for (auto& output : outputInfo) {
        output.second->setPrecision(InferenceEngine::Precision::FP32);
        output.second->setLayout(InferenceEngine::Layout::NCHW);
        (this->outputs).insert(std::pair<std::string, InferenceEngine::DataPtr&>(output.first, output.second));
        (this->outputsNames).push_back(output.first);
    }

    if (auto ngraphFunction = (this->cnnNetwork).getFunction()) {
        for (const auto op : ngraphFunction->get_ops()) {
            auto outputLayer = outputInfo.find(op->get_friendly_name());
            if (outputLayer != outputInfo.end()) {
                auto regionYolo = std::dynamic_pointer_cast<ngraph::op::RegionYolo>(op);
                if (!regionYolo) {
                    throw std::runtime_error("Invalid output type: " +
                        std::string(regionYolo->get_type_info().name) + ". RegionYolo expected");
                }
              
                this->params.insert(std::pair<std::string, Params*>(outputLayer->first,new Params(regionYolo)));
            }
        }
    }
    else {
        throw std::runtime_error("Can't get ngraph::Function. Make sure the provided model is in IR version 10 or greater.");
    }

    if (!(this->labels).empty() && static_cast<int>((this->labels).size()) != ((params.begin())->second)->classes) {
        throw std::runtime_error("The number of labels is different from numbers of model classes");
    }

}

void Yolov3::setConstInput(InferenceEngine::InferRequest::Ptr& inferReq) {}

void Yolov3::processOutput(std::map< std::string, InferenceEngine::Blob::Ptr>& outputs, cv::Mat frame, bool printOutput, double threshold) {

    // Processing results of the CURRENT request
    std::vector<DetectionObject> objects;
    // Parsing outputs
    for (auto& output : outputs) {
        this->parseYOLOV3Output(output.first, output.second, this->inputHeight, this->inputWidth, frame.size().height, frame.size().width, threshold, objects);
    }
    // Filtering overlapping boxes
    double FLAGS_iou_t=0;
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

template <typename T>
void Params::computeAnchors(const std::vector<T>& mask) {
    std::vector<float> maskedAnchors(num * 2);
    for (int i = 0; i < num; ++i) {
        maskedAnchors[i * 2] = anchors[mask[i] * 2];
        maskedAnchors[i * 2 + 1] = anchors[mask[i] * 2 + 1];
    }
    anchors = maskedAnchors;
}

Params::Params(const std::shared_ptr<ngraph::op::RegionYolo> regionYolo) {
    coords = regionYolo->get_num_coords();
    classes = regionYolo->get_num_classes();
    anchors = regionYolo->get_anchors();
    auto mask = regionYolo->get_mask();
    num = mask.size();

    computeAnchors(mask);
}

