// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine crossroad_camera demo application
* \file crossroad_camera_demo/main.cpp
* \example crossroad_camera_demo/main.cpp
*/
#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <map>
#include <string>
#include <vector>
#include <set>

#include "openvino/openvino.hpp"

#include "monitors/presenter.h"
#include "utils/images_capture.h"
#include "utils/ocv_common.hpp"
#include "utils/performance_metrics.hpp"
#include "utils/slog.hpp"
#include "crossroad_camera_demo.hpp"

using namespace ov::preprocess;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

// -------------------------Generic routines for detection networks-------------------------------------------------

struct BaseDetection {
    ov::runtime::ExecutableNetwork net;
    ov::runtime::InferRequest request;
    std::string& commandLineFlag;
    std::string topoName;
    ov::runtime::Tensor input_tensor;
    std::string inputName;
    std::string outputName;

    BaseDetection(std::string& commandLineFlag, const std::string& topoName)
            : commandLineFlag(commandLineFlag), topoName(topoName) {}

    ov::runtime::ExecutableNetwork* operator->() {
        return &net;
    }

    virtual std::shared_ptr<ov::Function> read(const ov::runtime::Core& core) = 0;

    virtual void setRoiBlob(const ov::runtime::Tensor& roi_tensor) {
        if (!enabled())
            return;
        if (!request)
            request = net.create_infer_request();

        request.set_input_tensor(roi_tensor);
    }

    virtual void enqueue(const cv::Mat& person) {
        if (!enabled())
            return;
        if (!request)
            request = net.create_infer_request();

        input_tensor = request.get_input_tensor();
        matToTensor(person, input_tensor);
    }

    virtual void submitRequest() {
        if (!enabled() || !request)
            return;

        request.start_async();
    }

    virtual void wait() {
        if (!enabled()|| !request)
            return;

        request.wait();
    }

    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                slog::info << topoName << " detection DISABLED" << slog::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
};

struct PersonDetection : BaseDetection {
    size_t maxProposalCount;
    size_t objectSize;
    float width = 0.0f;
    float height = 0.0f;
    bool resultsFetched = false;

    struct Result {
        int label = 0;
        float confidence = .0f;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void setRoiBlob(const ov::runtime::Tensor& roi_tensor) override {
        ov::Shape shape = roi_tensor.get_shape();
        ov::Layout layout("NHWC");

        height = static_cast<float>(roi_tensor.get_shape()[ov::layout::height_idx(layout)]);
        width = static_cast<float>(roi_tensor.get_shape()[ov::layout::width_idx(layout)]);
        BaseDetection::setRoiBlob(roi_tensor);
    }

    void enqueue(const cv::Mat& frame) override {
        height = static_cast<float>(frame.rows);
        width = static_cast<float>(frame.cols);
        BaseDetection::enqueue(frame);
    }

    PersonDetection() : BaseDetection(FLAGS_m, "Person Detection"), maxProposalCount(0), objectSize(0) {}

    std::shared_ptr<ov::Function> read(const ov::runtime::Core& core) override {
        /** Read network model **/
        auto network = core.read_model(FLAGS_m);
        /** Set batch size to 1 **/
        const ov::Layout layout_nhwc{"NHWC"};
        ov::Shape input_shape = network->input().get_shape();
        input_shape[ov::layout::batch_idx(layout_nhwc)] = 1;
        network->reshape({{network->input().get_any_name(), input_shape}});
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        ov::OutputVector inputs = network->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Person Detection network should have only one input");
        }

        inputName = network->input().get_any_name();
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        ov::OutputVector outputs = network->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("Person Detection network should have only one output");
        }

        outputName = network->output().get_any_name();

        maxProposalCount = network->output().get_shape()[2];
        objectSize = network->output().get_shape()[3];

        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputs[0].get_shape().size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }

// dynamic change for tensor shape not working yet
//        ov::Shape tensor_shape = inputs[0].get_shape();
//        tensor_shape[1] = 1080;
//        tensor_shape[2] = 1920;
        const ov::Layout tensor_layout{ "NHWC" };

        auto proc = PrePostProcessor(network);
        if (FLAGS_auto_resize) {
            proc.input().tensor().
                set_element_type(ov::element::u8).
//                                  set_spatial_static_shape(
//                                      tensor_shape[ov::layout::height_idx(tensor_layout)],
//                                      tensor_shape[ov::layout::width_idx(tensor_layout)]).
                set_spatial_dynamic_shape().
                set_layout(tensor_layout);
            proc.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW").
                resize(ResizeAlgorithm::RESIZE_LINEAR);
            proc.input().network().set_layout("NCHW");
            proc.output().tensor().set_element_type(ov::element::f32);
        } else {
            proc.input().tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }

        network = proc.build();

        return network;
    }

    void fetchResults() {
        if (!enabled())
            return;

        results.clear();

        if (resultsFetched)
            return;

        resultsFetched = true;
        const float* detections = request.get_output_tensor().data<float>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            if (image_id < 0) {  // indicates end of detections
                break;
            }

            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];

            r.location.x = static_cast<int>(detections[i * objectSize + 3] * width);
            r.location.y = static_cast<int>(detections[i * objectSize + 4] * height);
            r.location.width = static_cast<int>(detections[i * objectSize + 5] * width - r.location.x);
            r.location.height = static_cast<int>(detections[i * objectSize + 6] * height - r.location.y);

            if (r.confidence <= FLAGS_t) {
                continue;
            }

            if (FLAGS_r) {
                slog::debug << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                          << r.location.height << ")"
                          << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << slog::endl;
            }
            results.push_back(r);
        }
    }
};

struct PersonAttribsDetection : BaseDetection {
    std::string outputNameForAttributes;
    std::string outputNameForTopColorPoint;
    std::string outputNameForBottomColorPoint;
    bool hasTopBottomColor;


    PersonAttribsDetection() : BaseDetection(FLAGS_m_pa, "Person Attributes Recognition"), hasTopBottomColor(false) {}

    struct AttributesAndColorPoints {
        std::vector<std::string> attributes_strings;
        std::vector<bool> attributes_indicators;

        cv::Point2f top_color_point;
        cv::Point2f bottom_color_point;
        cv::Vec3b top_color;
        cv::Vec3b bottom_color;
    };

    static cv::Vec3b GetAvgColor(const cv::Mat& image) {
        int clusterCount = 5;
        cv::Mat labels;
        cv::Mat centers;
        cv::Mat image32f;
        image.convertTo(image32f, CV_32F);
        image32f = image32f.reshape(1, image32f.rows * image32f.cols);
        clusterCount = std::min(clusterCount, image32f.rows);
        cv::kmeans(image32f, clusterCount, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 10, 1.0),
                    10, cv::KMEANS_RANDOM_CENTERS, centers);
        centers.convertTo(centers, CV_8U);
        centers = centers.reshape(3, clusterCount);
        std::vector<int> freq(clusterCount);

        for (int i = 0; i < labels.rows * labels.cols; ++i) {
            freq[labels.at<int>(i)]++;
        }

        int freqArgmax = static_cast<int>(std::max_element(freq.begin(), freq.end()) - freq.begin());

        return centers.at<cv::Vec3b>(freqArgmax);
    }

    AttributesAndColorPoints GetPersonAttributes() {
        static const char *const attributeStringsFor7Attributes[] = {
                "is male", "has_bag", "has hat", "has longsleeves", "has longpants", "has longhair", "has coat_jacket"
        };
        static const char *const attributeStringsFor8Attributes[] = {
                "is male", "has_bag", "has_backpack" , "has hat", "has longsleeves", "has longpants", "has longhair", "has coat_jacket"
        };

        ov::runtime::Tensor attribsBlob = request.get_tensor(outputNameForAttributes);
        size_t numOfAttrChannels = attribsBlob.get_shape()[1];

        const char* const* attributeStrings;
        if (numOfAttrChannels == arraySize(attributeStringsFor7Attributes)) {
            attributeStrings = attributeStringsFor7Attributes;
        } else if (numOfAttrChannels == arraySize(attributeStringsFor8Attributes)) {
            attributeStrings = attributeStringsFor8Attributes;
        } else {
            throw std::logic_error("Output size (" + std::to_string(numOfAttrChannels) + ") of the "
                                   "Person Attributes Recognition network is not equal to expected "
                                   "number of attributes ("
                                   + std::to_string(arraySize(attributeStringsFor7Attributes))
                                   + " or "
                                   + std::to_string(arraySize(attributeStringsFor7Attributes)) + ")");
        }

        AttributesAndColorPoints returnValue;

        auto outputAttrValues = attribsBlob.data<float>();
        for (size_t i = 0; i < numOfAttrChannels; i++) {
            returnValue.attributes_strings.push_back(attributeStrings[i]);
            returnValue.attributes_indicators.push_back(outputAttrValues[i] > 0.5);
        }

        if (hasTopBottomColor) {
            ov::runtime::Tensor topColorPointBlob = request.get_tensor(outputNameForTopColorPoint);
            ov::runtime::Tensor bottomColorPointBlob = request.get_tensor(outputNameForBottomColorPoint);

            size_t numOfTCPointChannels = topColorPointBlob.get_shape()[1];
            size_t numOfBCPointChannels = bottomColorPointBlob.get_shape()[1];
            if (numOfTCPointChannels != 2) {
                throw std::logic_error("Output size (" + std::to_string(numOfTCPointChannels) + ") of the "
                                       "Person Attributes Recognition network is not equal to point coordinates(2)");
            }
            if (numOfBCPointChannels != 2) {
                throw std::logic_error("Output size (" + std::to_string(numOfBCPointChannels) + ") of the "
                                       "Person Attributes Recognition network is not equal to point coordinates (2)");
            }

            auto outputTCPointValues = topColorPointBlob.data<float>();
            auto outputBCPointValues = bottomColorPointBlob.data<float>();

            returnValue.top_color_point.x = outputTCPointValues[0];
            returnValue.top_color_point.y = outputTCPointValues[1];

            returnValue.bottom_color_point.x = outputBCPointValues[0];
            returnValue.bottom_color_point.y = outputBCPointValues[1];
        }

        return returnValue;
    }

    bool HasTopBottomColor() const {
        return hasTopBottomColor;
    }

    std::shared_ptr<ov::Function> read(const ov::runtime::Core& core) override {
        /** Read network model **/
        auto network = core.read_model(FLAGS_m_pa);
        /** Extract model name and load it's weights **/
        const ov::Layout layout_nhwc{ "NHWC" };
        ov::Shape input_shape = network->input().get_shape();
        input_shape[ov::layout::batch_idx(layout_nhwc)] = 1;
        network->reshape({ {network->input().get_any_name(), input_shape} });

        // -----------------------------------------------------------------------------------------------------

        /** Person Attribs network should have one input and one or three outputs **/
        // ---------------------------Check inputs ------------------------------------------------------
        ov::OutputVector inputs = network->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Person Attribs topology should have only one input");
        }

        auto proc = PrePostProcessor(network);
        if (FLAGS_auto_resize) {
            proc.input().tensor().
                set_element_type(ov::element::u8).
                set_spatial_dynamic_shape().
                set_layout({ "NHWC" });
            proc.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW").
                resize(ResizeAlgorithm::RESIZE_LINEAR);
            proc.input().network().set_layout("NCHW");
        } else {
            proc.input().tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }

        network = proc.build();

        inputName = network->input().get_any_name();
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        ov::OutputVector outputs = network->outputs();

        if (outputs.size() == 1) {
            // attribute probabilities for
            // person-attributes-recognition-crossroad-0234 and person-attributes-recognition-crossroad-0238 models
            outputNameForAttributes = outputs[0].get_any_name();
            hasTopBottomColor = false;
        } else if (outputs.size() == 3) {
            // attribute probabilities for person-attributes-recognition-crossroad-0230
            std::string o1 = "453";
            auto it1 = std::find_if(outputs.begin(), outputs.end(), [&o1](const ov::Output<ov::Node>& obj) {return obj.get_any_name() == o1;});
            if (it1 != outputs.end()) {
                outputNameForAttributes = it1->get_any_name();
            }
            std::string o2 = "456";
            auto it2 = std::find_if(outputs.begin(), outputs.end(), [&o2](const ov::Output<ov::Node>& obj) {return obj.get_any_name() == o2;});
            if (it2 != outputs.end()) {
                outputNameForTopColorPoint = it2->get_any_name();
            }
            std::string o3 = "459";
            auto it3 = std::find_if(outputs.begin(), outputs.end(), [&o3](const ov::Output<ov::Node>& obj) {return obj.get_any_name() == o3;});
            if (it3 != outputs.end()) {
                outputNameForBottomColorPoint = it3->get_any_name();
            }
            hasTopBottomColor = true;
        } else {
            throw std::logic_error("Person Attribs Network expects either a network having one output (person attributes), "
                "or a network having three outputs (person attributes, top color point, bottom color point)");
        }

        _enabled = true;

        return network;
    }
};

struct PersonReIdentification : BaseDetection {
    std::vector<std::vector<float>> globalReIdVec;  // contains vectors characterising all detected persons

    PersonReIdentification() : BaseDetection(FLAGS_m_reid, "Person Re-Identification Retail") {}

    unsigned long int findMatchingPerson(const std::vector<float>& newReIdVec) {
        auto size = globalReIdVec.size();

        /* assigned REID is index of the matched vector from the globalReIdVec */
        for (size_t i = 0; i < size; ++i) {
            float cosSim = cosineSimilarity(newReIdVec, globalReIdVec[i]);
            if (FLAGS_r) {
                slog::debug << "cosineSimilarity: " << cosSim << slog::endl;
            }
            if (cosSim > FLAGS_t_reid) {
                /* We substitute previous person's vector by a new one characterising
                 * last person's position */
                globalReIdVec[i] = newReIdVec;
                return i;
            }
        }
        globalReIdVec.push_back(newReIdVec);
        return size;
    }

    std::vector<float> getReidVec() {
        ov::runtime::Tensor attribsBlob = request.get_tensor(outputName);

        auto numOfChannels = attribsBlob.get_shape()[1];
        auto outputValues = attribsBlob.data<float>();

        return std::vector<float>(outputValues, outputValues + numOfChannels);
    }

    template <typename T>
    float cosineSimilarity(const std::vector<T>& vecA, const std::vector<T>& vecB) {
        if (vecA.size() != vecB.size()) {
            throw std::logic_error("cosine similarity can't be called for the vectors of different lengths: "
                                   "vecA size = " + std::to_string(vecA.size()) +
                                   "vecB size = " + std::to_string(vecB.size()));
        }

        T mul, denomA, denomB, A, B;
        mul = denomA = denomB = A = B = 0;
        for (size_t i = 0; i < vecA.size(); ++i) {
            A = vecA[i];
            B = vecB[i];
            mul += A * B;
            denomA += A * A;
            denomB += B * B;
        }
        if (denomA == 0 || denomB == 0) {
            throw std::logic_error("cosine similarity is not defined whenever one or both "
                                   "input vectors are zero-vectors.");
        }
        return mul / (sqrt(denomA) * sqrt(denomB));
    }

    std::shared_ptr<ov::Function> read(const ov::runtime::Core& core) override {
        /** Read network model **/
        auto network = core.read_model(FLAGS_m_reid);
        const ov::Layout layout_nhwc{ "NHWC" };
        ov::Shape input_shape = network->input().get_shape();
        input_shape[ov::layout::batch_idx(layout_nhwc)] = 1;
        network->reshape({ {network->input().get_any_name(), input_shape} });

        /** Person Reidentification network should have 1 input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        ov::OutputVector inputs = network->inputs();
        if (inputs.size() != 1) {
            throw std::logic_error("Person Reidentification Retail should have 1 input");
        }

        auto proc = PrePostProcessor(network);
        if (FLAGS_auto_resize) {
            proc.input().tensor().
                set_element_type(ov::element::u8).
                set_spatial_dynamic_shape().
                set_layout({ "NHWC" });
            proc.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW").
                resize(ResizeAlgorithm::RESIZE_LINEAR);
            proc.input().network().set_layout("NCHW");
        } else {
            proc.input().tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }

        network = proc.build();

        inputName = network->input().get_any_name();
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        ov::OutputVector outputs = network->outputs();
        if (outputs.size() != 1) {
            throw std::logic_error("Person Re-Identification Model should have 1 output");
        }

        outputName = network->output().get_any_name();

        _enabled = true;
        return network;
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) {}

    void into(ov::runtime::Core& core, const std::string& deviceName) const {
        if (detector.enabled()) {
            detector.net = core.compile_model(detector.read(core), deviceName);
            logExecNetworkInfo(detector.net, detector.commandLineFlag, deviceName, detector.topoName);
        }
    }
};



int main(int argc, char* argv[]) {
    try {
        PerformanceMetrics metrics;

        /** This demo covers 3 certain topologies and cannot be generalized **/
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::unique_ptr<ImagesCapture> cap = openImagesCapture(FLAGS_i, FLAGS_loop);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << ov::get_openvino_version() << slog::endl;

        ov::runtime::Core core;

        std::set<std::string> loadedDevices;

        PersonDetection personDetection;
        PersonAttribsDetection personAttribs;
        PersonReIdentification personReId;

        std::vector<std::string> deviceNames = {
            FLAGS_d,
            personAttribs.enabled() ? FLAGS_d_pa : "",
            personReId.enabled() ? FLAGS_d_reid : ""
        };

        for (auto&& flag : deviceNames) {
            if (flag.empty())
                continue;

            auto i = loadedDevices.find(flag);
            if (i != loadedDevices.end()) {
                continue;
            }

            if ((flag.find("CPU") != std::string::npos)) {
                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = std::make_shared<InferenceEngine::Extension>(FLAGS_l);
                    core.add_extension(extension_ptr);
                }
            }

            if ((flag.find("GPU") != std::string::npos) && !FLAGS_c.empty()) {
                // Load any user-specified clDNN Extensions
                core.set_config({ { InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } }, "GPU");
            }

            loadedDevices.insert(flag);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR models and load them to devices ------------------------------
        Load(personDetection).into(core, FLAGS_d);
        Load(personAttribs).into(core, FLAGS_d_pa);
        Load(personReId).into(core, FLAGS_d_reid);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Do inference ---------------------------------------------------------
        InferenceEngine::ROI cropRoi;  // cropped image coordinates
        ov::runtime::Tensor frameTensor;
        ov::runtime::Tensor roiTensor;
        cv::Mat person;  // Mat object containing person data cropped by openCV

        /** Start inference & calc performance **/
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

        auto startTime = std::chrono::steady_clock::now();
        cv::Mat frame = cap->read();
        if (!frame.data) {
            throw std::logic_error("Can't read an image from the input");
        }

        cv::VideoWriter videoWriter;
        if (!FLAGS_o.empty() && !videoWriter.open(FLAGS_o, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                  cap->fps(), frame.size())) {
            throw std::runtime_error("Can't open video writer");
        }
        uint32_t framesProcessed = 0;
        cv::Size graphSize{frame.cols / 4, 60};
        Presenter presenter(FLAGS_u, frame.rows - graphSize.height - 10, graphSize);

        bool shouldHandleTopBottomColors = personAttribs.HasTopBottomColor();

        do {
            if (FLAGS_auto_resize) {
                // just wrap Mat object with Tensor without additional memory allocation
                frameTensor = wrapMat2Tensor(frame);
                personDetection.setRoiBlob(frameTensor);
            } else {
                // resize Mat and copy data into OpenVINO allocated Tensor
                personDetection.enqueue(frame);
            }
            // --------------------------- Run Person detection inference --------------------------------------
            auto t0 = std::chrono::high_resolution_clock::now();
            personDetection.submitRequest();
            personDetection.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            ms detection = std::chrono::duration_cast<ms>(t1 - t0);
            // parse inference results internally (e.g. apply a threshold, etc)
            personDetection.fetchResults();
            // -------------------------------------------------------------------------------------------------

            // --------------------------- Process the results down to the pipeline ----------------------------
            ms personAttribsNetworkTime(0), personReIdNetworktime(0);
            int personAttribsInferred = 0,  personReIdInferred = 0;
            for (auto && result : personDetection.results) {
                if (result.label == FLAGS_person_label) {
                    // person
                    if (FLAGS_auto_resize) {
                        cropRoi.posX = (result.location.x < 0) ? 0 : result.location.x;
                        cropRoi.posY = (result.location.y < 0) ? 0 : result.location.y;
                        cropRoi.sizeX = std::min((size_t) result.location.width, frame.cols - cropRoi.posX);
                        cropRoi.sizeY = std::min((size_t) result.location.height, frame.rows - cropRoi.posY);
                        ov::Coordinate p00({ 0, cropRoi.posY, cropRoi.posX, 0 });
                        ov::Coordinate p01({ 1, cropRoi.posY + cropRoi.sizeY, cropRoi.posX + cropRoi.sizeX, 3 });
                        roiTensor = ov::runtime::Tensor(frameTensor, p00, p01);
                    } else {
                        // To crop ROI manually and allocate required memory (cv::Mat) again
                        auto clippedRect = result.location & cv::Rect(0, 0, frame.cols, frame.rows);
                        person = frame(clippedRect);
                    }

                    PersonAttribsDetection::AttributesAndColorPoints resPersAttrAndColor;
                    if (personAttribs.enabled()) {
                        // --------------------------- Run Person Attributes Recognition -----------------------
                        if (FLAGS_auto_resize) {
                            personAttribs.setRoiBlob(roiTensor);
                        } else {
                            personAttribs.enqueue(person);
                        }

                        t0 = std::chrono::high_resolution_clock::now();
                        personAttribs.submitRequest();
                        personAttribs.wait();
                        t1 = std::chrono::high_resolution_clock::now();
                        personAttribsNetworkTime += std::chrono::duration_cast<ms>(t1 - t0);
                        personAttribsInferred++;
                        // --------------------------- Process outputs -----------------------------------------

                        resPersAttrAndColor = personAttribs.GetPersonAttributes();

                        if (shouldHandleTopBottomColors) {
                            cv::Point top_color_p;
                            cv::Point bottom_color_p;

                            top_color_p.x = static_cast<int>(resPersAttrAndColor.top_color_point.x) * person.cols;
                            top_color_p.y = static_cast<int>(resPersAttrAndColor.top_color_point.y) * person.rows;

                            bottom_color_p.x = static_cast<int>(resPersAttrAndColor.bottom_color_point.x) * person.cols;
                            bottom_color_p.y = static_cast<int>(resPersAttrAndColor.bottom_color_point.y) * person.rows;


                            cv::Rect person_rect(0, 0, person.cols, person.rows);

                            // Define area around top color's location
                            cv::Rect tc_rect;
                            tc_rect.x = top_color_p.x - person.cols / 6;
                            tc_rect.y = top_color_p.y - person.rows / 10;
                            tc_rect.height = 2 * person.rows / 8;
                            tc_rect.width = 2 * person.cols / 6;

                            tc_rect = tc_rect & person_rect;

                            // Define area around bottom color's location
                            cv::Rect bc_rect;
                            bc_rect.x = bottom_color_p.x - person.cols / 6;
                            bc_rect.y = bottom_color_p.y - person.rows / 10;
                            bc_rect.height =  2 * person.rows / 8;
                            bc_rect.width = 2 * person.cols / 6;

                            bc_rect = bc_rect & person_rect;

                            if (!tc_rect.empty())
                                resPersAttrAndColor.top_color = PersonAttribsDetection::GetAvgColor(person(tc_rect));
                            if (!bc_rect.empty())
                                resPersAttrAndColor.bottom_color = PersonAttribsDetection::GetAvgColor(person(bc_rect));
                        }
                    }

                    std::string resPersReid = "";
                    if (personReId.enabled()) {
                        // --------------------------- Run Person Reidentification -----------------------------
                        if (FLAGS_auto_resize) {
                            personReId.setRoiBlob(roiTensor);
                        } else {
                            personReId.enqueue(person);
                        }

                        t0 = std::chrono::high_resolution_clock::now();
                        personReId.submitRequest();
                        personReId.wait();
                        t1 = std::chrono::high_resolution_clock::now();

                        personReIdNetworktime += std::chrono::duration_cast<ms>(t1 - t0);
                        personReIdInferred++;

                        auto reIdVector = personReId.getReidVec();

                        /* Check cosine similarity with all previously detected persons.
                           If it's new person it is added to the global Reid vector and
                           new global ID is assigned to the person. Otherwise, ID of
                           matched person is assigned to it. */
                        auto foundId = personReId.findMatchingPerson(reIdVector);
                        resPersReid = "REID: " + std::to_string(foundId);
                    }

                    // --------------------------- Process outputs -----------------------------------------
                    if (!resPersAttrAndColor.attributes_strings.empty()) {
                        cv::Rect image_area(0, 0, frame.cols, frame.rows);
                        cv::Rect tc_label(result.location.x + result.location.width, result.location.y,
                                          result.location.width / 4, result.location.height / 2);
                        cv::Rect bc_label(result.location.x + result.location.width, result.location.y + result.location.height / 2,
                                            result.location.width / 4, result.location.height / 2);

                        if (shouldHandleTopBottomColors) {
                            frame(tc_label & image_area) = resPersAttrAndColor.top_color;
                            frame(bc_label & image_area) = resPersAttrAndColor.bottom_color;
                        }

                        for (size_t i = 0; i < resPersAttrAndColor.attributes_strings.size(); ++i) {
                            cv::Scalar color;
                            if (resPersAttrAndColor.attributes_indicators[i]) {
                                color = cv::Scalar(0, 200, 0); // has attribute
                            } else {
                                color = cv::Scalar(0, 0, 255); // doesn't have attribute
                            }
                            putHighlightedText(frame,
                                    resPersAttrAndColor.attributes_strings[i],
                                    cv::Point2f(static_cast<float>(result.location.x + 5 * result.location.width / 4),
                                                static_cast<float>(result.location.y + 15 + 15 * i)),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.5,
                                    color, 1);
                        }

                        if (FLAGS_r) {
                            std::string output_attribute_string;
                            for (size_t i = 0; i < resPersAttrAndColor.attributes_strings.size(); ++i)
                                if (resPersAttrAndColor.attributes_indicators[i])
                                    output_attribute_string += resPersAttrAndColor.attributes_strings[i] + ",";
                            slog::debug << "Person Attributes results: " << output_attribute_string << slog::endl;
                            if (shouldHandleTopBottomColors) {
                                slog::debug << "Person top color: " << resPersAttrAndColor.top_color << slog::endl;
                                slog::debug << "Person bottom color: " << resPersAttrAndColor.bottom_color << slog::endl;
                            }
                        }
                    }
                    if (!resPersReid.empty()) {
                        putHighlightedText(frame,
                                    resPersReid,
                                    cv::Point2f(static_cast<float>(result.location.x), static_cast<float>(result.location.y + 30)),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.55,
                                    cv::Scalar(250, 10, 10), 1);

                        if (FLAGS_r) {
                            slog::debug << "Person Re-Identification results: " << resPersReid << slog::endl;
                        }
                    }
                    cv::rectangle(frame, result.location, cv::Scalar(0, 255, 0), 1);
                }
            }

            presenter.drawGraphs(frame);
            metrics.update(startTime);

            // --------------------------- Execution statistics ------------------------------------------------
            std::ostringstream out;
            out << "Detection time : " << std::fixed << std::setprecision(2) << detection.count()
                << " ms (" << 1000.f / detection.count() << " fps)";

            putHighlightedText(frame, out.str(), cv::Point2f(0, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, { 200, 10, 10 }, 2);

            if (personDetection.results.size()) {
                if (personAttribs.enabled() && personAttribsInferred) {
                    float average_time = static_cast<float>(personAttribsNetworkTime.count() / personAttribsInferred);
                    out.str("");
                    out << "Attributes Recognition time: " << std::fixed << std::setprecision(2) << average_time
                        << " ms (" << 1000.f / average_time << " fps)";
                    putHighlightedText(frame, out.str(), cv::Point2f(0, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, { 200, 10, 10 }, 2);
                    if (FLAGS_r) {
                        slog::debug << out.str() << slog::endl;
                    }
                }
                if (personReId.enabled() && personReIdInferred) {
                    float average_time = static_cast<float>(personReIdNetworktime.count() / personReIdInferred);
                    out.str("");
                    out << "Re-Identification time: " << std::fixed << std::setprecision(2) << average_time
                        << " ms (" << 1000.f / average_time << " fps)";
                    putHighlightedText(frame, out.str(), cv::Point2f(0, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, { 200, 10, 10 }, 2);
                    if (FLAGS_r) {
                        slog::debug << out.str() << slog::endl;
                    }
                }
            }
            framesProcessed++;
            if (videoWriter.isOpened() && (FLAGS_limit == 0 || framesProcessed <= FLAGS_limit)) {
                videoWriter.write(frame);
            }
            if (!FLAGS_no_show) {
                cv::imshow("Detection results", frame);
                const int key = cv::waitKey(1);
                if (27 == key)  // Esc
                    break;
                presenter.handleKey(key);
            }
            startTime = std::chrono::steady_clock::now();
            frame = cap->read();
        } while (frame.data);

        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception& error) {
        slog ::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    return 0;
}
