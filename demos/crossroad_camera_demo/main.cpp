// Copyright (C) 2018 Intel Corporation
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

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include "crossroad_camera_demo.hpp"
#include <ext_list.hpp>

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    std::cout << "[ INFO ] Parsing input parameters" << std::endl;

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
    ExecutableNetwork net;
    InferencePlugin plugin;
    InferRequest request;
    std::string & commandLineFlag;
    std::string topoName;
    Blob::Ptr inputBlob;
    std::string inputName;
    std::string outputName;

    BaseDetection(std::string &commandLineFlag, std::string topoName)
            : commandLineFlag(commandLineFlag), topoName(topoName) {}

    ExecutableNetwork * operator ->() {
        return &net;
    }
    virtual CNNNetwork read()  = 0;

    virtual void setRoiBlob(const Blob::Ptr &roiBlob) {
        if (!enabled())
            return;
        if (!request)
            request = net.CreateInferRequest();

        request.SetBlob(inputName, roiBlob);
    }

    virtual void enqueue(const cv::Mat &person) {
        if (!enabled())
            return;
        if (!request)
            request = net.CreateInferRequest();

        if (FLAGS_auto_resize) {
            inputBlob = wrapMat2Blob(person);
            request.SetBlob(inputName, inputBlob);
        } else {
            inputBlob = request.GetBlob(inputName);
            matU8ToBlob<uint8_t>(person, inputBlob);
        }
    }

    virtual void submitRequest() {
        if (!enabled() || !request) return;
        request.StartAsync();
    }

    virtual void wait() {
        if (!enabled()|| !request) return;
        request.Wait(IInferRequest::WaitMode::RESULT_READY);
    }
    mutable bool enablingChecked = false;
    mutable bool _enabled = false;

    bool enabled() const  {
        if (!enablingChecked) {
            _enabled = !commandLineFlag.empty();
            if (!_enabled) {
                std::cout << "[ INFO ] " << topoName << " detection DISABLED" << std::endl;
            }
            enablingChecked = true;
        }
        return _enabled;
    }
};

struct PersonDetection : BaseDetection{
    int maxProposalCount;
    int objectSize;
    float width = 0;
    float height = 0;
    bool resultsFetched = false;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    std::vector<Result> results;

    void submitRequest() override {
        resultsFetched = false;
        results.clear();
        BaseDetection::submitRequest();
    }

    void setRoiBlob(const Blob::Ptr &frameBlob) override {
        height = frameBlob->getTensorDesc().getDims()[2];
        width = frameBlob->getTensorDesc().getDims()[3];
        BaseDetection::setRoiBlob(frameBlob);
    }

    void enqueue(const cv::Mat &frame) override {
        height = frame.rows;
        width = frame.cols;
        BaseDetection::enqueue(frame);
    }

    PersonDetection() : BaseDetection(FLAGS_m, "Person Detection") {}
    CNNNetwork read() override {
        std::cout << "[ INFO ] Loading network files for PersonDetection" << std::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        std::cout << "[ INFO ] Batch size is forced to  1" << std::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking Person Detection inputs" << std::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Person Detection network should have only one input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);

        if (FLAGS_auto_resize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        }
        inputName = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking Person Detection outputs" << std::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Person Detection network should have only one output");
        }
        DataPtr& _output = outputInfo.begin()->second;
        const SizeVector outputDims = _output->getTensorDesc().getDims();
        outputName = outputInfo.begin()->first;
        maxProposalCount = outputDims[2];
        objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        std::cout << "[ INFO ] Loading Person Detection model to the "<< FLAGS_d << " plugin" << std::endl;
        return netReader.getNetwork();
    }

    void fetchResults() {
        if (!enabled()) return;
        results.clear();
        if (resultsFetched) return;
        resultsFetched = true;
        const float *detections = request.GetBlob(outputName)->buffer().as<float *>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            if (image_id < 0) {  // indicates end of detections
                break;
            }

            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];

            r.location.x = detections[i * objectSize + 3] * width;
            r.location.y = detections[i * objectSize + 4] * height;
            r.location.width = detections[i * objectSize + 5] * width - r.location.x;
            r.location.height = detections[i * objectSize + 6] * height - r.location.y;

            if (FLAGS_r) {
                std::cout << "[" << i << "," << r.label << "] element, prob = " << r.confidence <<
                          "    (" << r.location.x << "," << r.location.y << ")-(" << r.location.width << ","
                          << r.location.height << ")"
                          << ((r.confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
            }

            if (r.confidence <= FLAGS_t) {
                continue;
            }
            results.push_back(r);
        }
    }
};

struct PersonAttribsDetection : BaseDetection {
    std::string outputNameForAttributes;
    std::string outputNameForTopColorPoint;
    std::string outputNameForBottomColorPoint;


    PersonAttribsDetection() : BaseDetection(FLAGS_m_pa, "Person Attributes Recognition") {}

    struct AttributesAndColorPoints{
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
        image32f = image32f.reshape(1, image32f.rows*image32f.cols);
        cv::kmeans(image32f, clusterCount, labels, cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 10, 1.0),
                    10, cv::KMEANS_RANDOM_CENTERS, centers);
        centers.convertTo(centers, CV_8U);
        centers = centers.reshape(0, clusterCount);
        std::map<int, cv::Vec3b, std::greater<int>> max_color;
        std::vector<int> freq(clusterCount);

        for (size_t i = 0; i < labels.rows * labels.cols; ++i) {
            freq[labels.at<int>(i)]++;
        }

        for (size_t i = 0; i < freq.size(); ++i) {
            max_color[freq[i]] = centers.at<cv::Vec3b>(i);
        }

        return max_color.begin()->second;
    }

    AttributesAndColorPoints GetPersonAttributes() {
        static const std::vector<std::string> attributesVec = {
                "is male", "has_bag", "has_backpack" , "has hat", "has longsleeves", "has longpants", "has longhair", "has coat_jacket"
        };

        Blob::Ptr attribsBlob = request.GetBlob(outputNameForAttributes);
        Blob::Ptr topColorPointBlob = request.GetBlob(outputNameForTopColorPoint);
        Blob::Ptr bottomColorPointBlob = request.GetBlob(outputNameForBottomColorPoint);
        int numOfAttrChannels = attribsBlob->getTensorDesc().getDims().at(1);
        int numOfTCPointChannels = topColorPointBlob->getTensorDesc().getDims().at(1);
        int numOfBCPointChannels = bottomColorPointBlob->getTensorDesc().getDims().at(1);

        if (numOfAttrChannels != attributesVec.size()) {
            throw std::logic_error("Output size (" + std::to_string(numOfAttrChannels) + ") of the "
                                   "Person Attributes Recognition network is not equal to used person "
                                   "attributes vector size (" + std::to_string(attributesVec.size()) + ")");
        }
        if (numOfTCPointChannels != 2) {
            throw std::logic_error("Output size (" + std::to_string(numOfTCPointChannels) + ") of the "
                                   "Person Attributes Recognition network is not equal to point coordinates(2)");
        }
        if (numOfBCPointChannels != 2) {
            throw std::logic_error("Output size (" + std::to_string(numOfBCPointChannels) + ") of the "
                                   "Person Attributes Recognition network is not equal to point coordinates (2)");
        }

        auto outputAttrValues = attribsBlob->buffer().as<float*>();
        auto outputTCPointValues = topColorPointBlob->buffer().as<float*>();
        auto outputBCPointValues = bottomColorPointBlob->buffer().as<float*>();

        AttributesAndColorPoints returnValue;

        returnValue.top_color_point.x = outputTCPointValues[0];
        returnValue.top_color_point.y = outputTCPointValues[1];

        returnValue.bottom_color_point.x = outputBCPointValues[0];
        returnValue.bottom_color_point.y = outputBCPointValues[1];

        for (int i = 0; i < attributesVec.size(); i++) {
            returnValue.attributes_strings.push_back(attributesVec[i]);
            returnValue.attributes_indicators.push_back(outputAttrValues[i] > 0.5);
        }

        return returnValue;
    }

    CNNNetwork read() override {
        std::cout << "[ INFO ] Loading network files for PersonAttribs" << std::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_pa);
        netReader.getNetwork().setBatchSize(1);
        std::cout << "[ INFO ] Batch size is forced to 1 for Person Attribs" << std::endl;

        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_pa) + ".bin";
        netReader.ReadWeights(binFileName);
        // -----------------------------------------------------------------------------------------------------

        /** Person Attribs network should have one input two outputs **/
        // ---------------------------Check inputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking PersonAttribs inputs" << std::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Person Attribs topology should have only one input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        }
        inputName = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking Person Attribs outputs" << std::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 3) {
             throw std::logic_error("Person Attribs Network expects networks having one output");
        }
        auto it = outputInfo.begin();
        outputNameForAttributes = (it++)->second->name;  // attribute probabilities
        outputNameForTopColorPoint = (it++)->second->name;  // top color location
        outputNameForBottomColorPoint = (it++)->second->name;  // bottom color location
        std::cout << "[ INFO ] Loading Person Attributes Recognition model to the "<< FLAGS_d_pa << " plugin" << std::endl;
        _enabled = true;
        return netReader.getNetwork();
    }
};

struct PersonReIdentification : BaseDetection {
    std::vector<std::vector<float>> globalReIdVec;  // contains vectors characterising all detected persons

    PersonReIdentification() : BaseDetection(FLAGS_m_reid, "Person Reidentification Retail") {}

    unsigned long int findMatchingPerson(const std::vector<float> &newReIdVec) {
        float cosSim;
        auto size = globalReIdVec.size();

        /* assigned REID is index of the matched vector from the globalReIdVec */
        for (auto i = 0; i < size; ++i) {
            cosSim = cosineSimilarity(newReIdVec, globalReIdVec[i]);
            if (FLAGS_r) {
                std::cout << "cosineSimilarity: " << cosSim << std::endl;
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
        Blob::Ptr attribsBlob = request.GetBlob(outputName);

        auto numOfChannels = attribsBlob->getTensorDesc().getDims().at(1);
        /* output descriptor of Person Reidentification Recognition network has size 256 */
        if (numOfChannels != 256) {
            throw std::logic_error("Output size (" + std::to_string(numOfChannels) + ") of the "
                                   "Person Reidentification network is not equal to 256");
        }

        auto outputValues = attribsBlob->buffer().as<float*>();
        return std::vector<float>(outputValues, outputValues + 256);
    }

    template <typename T>
    float cosineSimilarity(const std::vector<T> &vecA, const std::vector<T> &vecB) {
        if (vecA.size() != vecB.size()) {
            throw std::logic_error("cosine similarity can't be called for the vectors of different lengths: "
                                   "vecA size = " + std::to_string(vecA.size()) +
                                   "vecB size = " + std::to_string(vecB.size()));
        }

        T mul, denomA, denomB, A, B;
        mul = denomA = denomB = A = B = 0;
        for (auto i = 0; i < vecA.size(); ++i) {
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

    CNNNetwork read() override {
        std::cout << "[ INFO ] Loading network files for Person Reidentification" << std::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m_reid);
        std::cout << "[ INFO ] Batch size is forced to  1 for Person Reidentification Network" << std::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m_reid) + ".bin";
        netReader.ReadWeights(binFileName);

        /** Person Reidentification network should have 1 input and one output **/
        // ---------------------------Check inputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking Person Reidentification Network input" << std::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Person Reidentification Retail should have 1 input");
        }
        InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);
        if (FLAGS_auto_resize) {
            inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
            inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        } else {
            inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        }
        inputName = inputInfo.begin()->first;
        // -----------------------------------------------------------------------------------------------------

        // ---------------------------Check outputs ------------------------------------------------------
        std::cout << "[ INFO ] Checking Person Reidentification Network output" << std::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("Person Reidentification Network should have 1 output");
        }
        outputName = outputInfo.begin()->first;
        std::cout << "[ INFO ] Loading Person Reidentification Retail model to the "<< FLAGS_d_reid << " plugin" << std::endl;

        _enabled = true;
        return netReader.getNetwork();
    }
};

struct Load {
    BaseDetection& detector;
    explicit Load(BaseDetection& detector) : detector(detector) { }

    void into(InferencePlugin & plg) const {
        if (detector.enabled()) {
            detector.net = plg.LoadNetwork(detector.read(), {});
            detector.plugin = plg;
        }
    }
};



int main(int argc, char *argv[]) {
    try {
        /** This demo covers 3 certain topologies and cannot be generalized **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        std::cout << "[ INFO ] Reading input" << std::endl;
        cv::Mat frame = cv::imread(FLAGS_i, cv::IMREAD_COLOR);
        const bool isVideo = frame.empty();
        cv::VideoCapture cap;
        if (isVideo && !(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = isVideo ? (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH) : frame.size().width;
        const size_t height = isVideo ? (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT) : frame.size().height;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        std::map<std::string, InferencePlugin> pluginsForNetworks;
        std::vector<std::string> pluginNames = {
                FLAGS_d, FLAGS_d_pa, FLAGS_d_reid
        };

        PersonDetection personDetection;
        PersonAttribsDetection personAttribs;
        PersonReIdentification personReId;

        for (auto && flag : pluginNames) {
            if (flag == "") continue;
            auto i = pluginsForNetworks.find(flag);
            if (i != pluginsForNetworks.end()) {
                continue;
            }
            std::cout << "[ INFO ] Loading plugin " << flag << std::endl;
            InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(flag);

            /** Printing plugin version **/
            printPluginVersion(plugin, std::cout);

            if ((flag.find("CPU") != std::string::npos)) {
                /** Load default extensions lib for the CPU plugin (e.g. SSD's DetectionOutput)**/
                plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    plugin.AddExtension(extension_ptr);
                    std::cout << "CPU Extension loaded: " << FLAGS_l << std::endl;
                }
            }

            if ((flag.find("GPU") != std::string::npos) && !FLAGS_c.empty()) {
                // Load any user-specified clDNN Extensions
                plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            }
            pluginsForNetworks[flag] = plugin;
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForNetworks) {
                plugin.second.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR models and load them to plugins ------------------------------
        Load(personDetection).into(pluginsForNetworks[FLAGS_d]);
        Load(personAttribs).into(pluginsForNetworks[FLAGS_d_pa]);
        Load(personReId).into(pluginsForNetworks[FLAGS_d_reid]);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Do inference ---------------------------------------------------------
        Blob::Ptr frameBlob;  // Blob to be used to keep processed frame data
        ROI cropRoi;  // cropped image coordinates
        Blob::Ptr roiBlob;  // This blob contains data from cropped image (vehicle or license plate)
        cv::Mat person;  // Mat object containing person data cropped by openCV

        /** Start inference & calc performance **/
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        std::cout << "[ INFO ] Start inference " << std::endl;
        do {
            // get and enqueue the next frame (in case of video)
            if (isVideo && !cap.read(frame)) {
                if (frame.empty())
                    break;  // end of video file
                throw std::logic_error("Failed to get frame from cv::VideoCapture");
            }
            if (FLAGS_auto_resize) {
                // just wrap Mat object with Blob::Ptr without additional memory allocation
                frameBlob = wrapMat2Blob(frame);
                personDetection.setRoiBlob(frameBlob);
            } else {
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
                if (result.label == 1) {  // person
                    if (FLAGS_auto_resize) {
                        cropRoi.posX = (result.location.x < 0) ? 0 : result.location.x;
                        cropRoi.posY = (result.location.y < 0) ? 0 : result.location.y;
                        cropRoi.sizeX = std::min((size_t) result.location.width, width - cropRoi.posX);
                        cropRoi.sizeY = std::min((size_t) result.location.height, height - cropRoi.posY);
                        roiBlob = make_shared_blob(frameBlob, cropRoi);
                    } else {
                        // To crop ROI manually and allocate required memory (cv::Mat) again
                        auto clippedRect = result.location & cv::Rect(0, 0, width, height);
                        person = frame(clippedRect);
                    }
                    PersonAttribsDetection::AttributesAndColorPoints resPersAttrAndColor;
                    std::string resPersReid = "";
                    cv::Point top_color_p;
                    cv::Point bottom_color_p;

                    if (personAttribs.enabled()) {
                        // --------------------------- Run Person Attributes Recognition -----------------------
                        if (FLAGS_auto_resize) {
                            personAttribs.setRoiBlob(roiBlob);
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

                        top_color_p.x = resPersAttrAndColor.top_color_point.x * person.cols;
                        top_color_p.y = resPersAttrAndColor.top_color_point.y * person.rows;

                        bottom_color_p.x = resPersAttrAndColor.bottom_color_point.x * person.cols;
                        bottom_color_p.y = resPersAttrAndColor.bottom_color_point.y * person.rows;


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

                        resPersAttrAndColor.top_color = PersonAttribsDetection::GetAvgColor(person(tc_rect));
                        resPersAttrAndColor.bottom_color = PersonAttribsDetection::GetAvgColor(person(bc_rect));
                    }
                    if (personReId.enabled()) {
                        // --------------------------- Run Person Reidentification -----------------------------
                        if (FLAGS_auto_resize) {
                            personReId.setRoiBlob(roiBlob);
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

                        frame(tc_label & image_area) = resPersAttrAndColor.top_color;
                        frame(bc_label & image_area) = resPersAttrAndColor.bottom_color;

                        for (size_t i = 0; i < resPersAttrAndColor.attributes_strings.size(); ++i) {
                            cv::Scalar color;
                            if (resPersAttrAndColor.attributes_indicators[i]) {
                                color = cv::Scalar(0, 255, 0);
                            } else {
                                color = cv::Scalar(0, 0, 255);
                            }
                            cv::putText(frame,
                                    resPersAttrAndColor.attributes_strings[i],
                                    cv::Point2f(result.location.x + 5 * result.location.width / 4, result.location.y + 15 + 15 * i),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.5,
                                    color);
                        }

                        if (FLAGS_r) {
                            std::string output_attribute_string;
                            for (size_t i = 0; i < resPersAttrAndColor.attributes_strings.size(); ++i)
                                if (resPersAttrAndColor.attributes_indicators[i])
                                    output_attribute_string += resPersAttrAndColor.attributes_strings[i] + ",";
                            std::cout << "Person Attributes results: " << output_attribute_string << std::endl;
                            std::cout << "Person top color: " << resPersAttrAndColor.top_color << std::endl;
                            std::cout << "Person bottom color: " << resPersAttrAndColor.bottom_color << std::endl;
                        }
                    }
                    if (!resPersReid.empty()) {
                        cv::putText(frame,
                                    resPersReid,
                                    cv::Point2f(result.location.x, result.location.y + 30),
                                    cv::FONT_HERSHEY_COMPLEX_SMALL,
                                    0.6,
                                    cv::Scalar(255, 255, 255));

                        if (FLAGS_r) {
                            std::cout << "Person Reidentification results:" << resPersReid << std::endl;
                        }
                    }
                    cv::rectangle(frame, result.location, cv::Scalar(0, 255, 0), 1);
                }
            }

            // --------------------------- Execution statistics ------------------------------------------------
            std::ostringstream out;
            out << "Person detection time  : " << std::fixed << std::setprecision(2) << detection.count()
                << " ms ("
                << 1000.f / detection.count() << " fps)";
            cv::putText(frame, out.str(), cv::Point2f(0, 20), cv::FONT_HERSHEY_TRIPLEX, 0.5,
                        cv::Scalar(255, 0, 0));
            if (personDetection.results.size()) {
                if (personAttribs.enabled() && personAttribsInferred) {
                    float average_time = personAttribsNetworkTime.count() / personAttribsInferred;
                    out.str("");
                    out << "Person Attributes Recognition time (averaged over " << personAttribsInferred
                        << " detections) :" << std::fixed << std::setprecision(2) << average_time
                        << " ms " << "(" << 1000.f / average_time << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 0, 0));
                    if (FLAGS_r) {
                        std::cout << out.str() << std::endl;;
                    }
                }
                if (personReId.enabled() && personReIdInferred) {
                    float average_time = personReIdNetworktime.count() / personReIdInferred;
                    out.str("");
                    out << "Person Reidentification time (averaged over " << personReIdInferred
                        << " detections) :" << std::fixed << std::setprecision(2) << average_time
                        << " ms " << "(" << 1000.f / average_time << " fps)";
                    cv::putText(frame, out.str(), cv::Point2f(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 0, 0));
                    if (FLAGS_r) {
                        std::cout << out.str() << std::endl;;
                    }
                }
            }

            if (!FLAGS_no_show) {
                cv::imshow("Detection results", frame);
            }
            // for still images wait until any key is pressed, for video 1 ms is enough per frame
            const int key = cv::waitKey(isVideo ? 1 : 0);
            if (27 == key)  // Esc
                break;
        } while (isVideo);

        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "[ INFO ] Total Inference time: " << total.count() << std::endl;

        /** Show performace results **/
        if (FLAGS_pc) {
            for (auto && plugin : pluginsForNetworks) {
                std::cout << "[ INFO ] Performance counts for " << plugin.first << " plugin";
                printPerformanceCountsPlugin(plugin.second, std::cout);
            }
        }
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    std::cout << "[ INFO ] Execution successful" << std::endl;
    return 0;
}
