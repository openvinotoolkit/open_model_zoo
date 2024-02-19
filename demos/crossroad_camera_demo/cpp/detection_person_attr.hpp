// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/openvino.hpp"

#include "gflags/gflags.h"
#include "utils/slog.hpp"
#include "detection_base.hpp"

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
        static const char* const attributeStringsFor7Attributes[] = {
                "is male", "has_bag", "has hat", "has longsleeves", "has longpants", "has longhair", "has coat_jacket"
        };
        static const char* const attributeStringsFor8Attributes[] = {
                "is male", "has_bag", "has_backpack" , "has hat", "has longsleeves", "has longpants", "has longhair", "has coat_jacket"
        };

        ov::Tensor attribsTensor = m_infer_request.get_tensor(outputNameForAttributes);
        size_t numOfAttrChannels = attribsTensor.get_shape()[1];

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

        auto outputAttrValues = attribsTensor.data<float>();
        for (size_t i = 0; i < numOfAttrChannels; i++) {
            returnValue.attributes_strings.push_back(attributeStrings[i]);
            returnValue.attributes_indicators.push_back(outputAttrValues[i] > 0.5);
        }

        if (hasTopBottomColor) {
            ov::Tensor topColorPointTensor = m_infer_request.get_tensor(outputNameForTopColorPoint);
            ov::Tensor bottomColorPointTensor = m_infer_request.get_tensor(outputNameForBottomColorPoint);

            size_t numOfTCPointChannels = topColorPointTensor.get_shape()[1];
            size_t numOfBCPointChannels = bottomColorPointTensor.get_shape()[1];
            if (numOfTCPointChannels != 2) {
                throw std::logic_error("Output size (" + std::to_string(numOfTCPointChannels) + ") of the "
                                       "Person Attributes Recognition network is not equal to point coordinates(2)");
            }
            if (numOfBCPointChannels != 2) {
                throw std::logic_error("Output size (" + std::to_string(numOfBCPointChannels) + ") of the "
                                       "Person Attributes Recognition network is not equal to point coordinates (2)");
            }

            auto outputTCPointValues = topColorPointTensor.data<float>();
            auto outputBCPointValues = bottomColorPointTensor.data<float>();

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

    bool CheckOutputNameExist(const ov::OutputVector& outputs, const std::string name) {
        if (std::find_if(outputs.begin(), outputs.end(),
            [&](const ov::Output<ov::Node>& output) {return output.get_any_name() == name; }) == outputs.end()) {
            return false;
        }
        return true;
    }

    std::shared_ptr<ov::Model> read(const ov::Core& core) override {
        // Read network model
        slog::info << "Reading model: " << FLAGS_m_pa << slog::endl;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m_pa);
        logBasicModelInfo(model);

        // set batch size 1
        model->get_parameters()[0]->set_layout("NCHW");
        ov::set_batch(model, 1);

        // Person Attribs network should have one input and one or three outputs
        // Check inputs
        if (model->inputs().size() != 1) {
            throw std::logic_error("Person Attribs topology should have only one input");
        }

        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

        if (FLAGS_auto_resize) {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_spatial_dynamic_shape().
                set_layout({ "NHWC" });
            ppp.input().preprocess().
                convert_element_type(ov::element::f32).
                convert_layout("NCHW").
                resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
            ppp.input().model().set_layout("NCHW");
        } else {
            ppp.input().tensor().
                set_element_type(ov::element::u8).
                set_layout({ "NCHW" });
        }

        model = ppp.build();

        m_inputName = model->input().get_any_name();

        // Check outputs
        ov::OutputVector outputs = model->outputs();

        if (outputs.size() == 1) {
            // attribute probabilities for
            // person-attributes-recognition-crossroad-0234 and person-attributes-recognition-crossroad-0238 models
            outputNameForAttributes = outputs[0].get_any_name();
            hasTopBottomColor = false;
        } else if (outputs.size() == 3) {
            // check person-attributes-recognition-crossroad-0230 outputs exist
            outputNameForAttributes = "453";
            outputNameForTopColorPoint = "456";
            outputNameForBottomColorPoint = "459";
            if (!CheckOutputNameExist(outputs, outputNameForAttributes)) {
                throw std::logic_error(std::string("Couldn't find output with name ") + outputNameForAttributes);
            }

            if (!CheckOutputNameExist(outputs, outputNameForTopColorPoint)) {
                throw std::logic_error(std::string("Couldn't find output with name ") + outputNameForTopColorPoint);
            }

            if (!CheckOutputNameExist(outputs, outputNameForBottomColorPoint)) {
                throw std::logic_error(std::string("Couldn't find output with name ") + outputNameForBottomColorPoint);
            }
            hasTopBottomColor = true;
        } else {
            throw std::logic_error("Person Attribs Network expects either a network having one output (person attributes), "
                                   "or a network having three outputs (person attributes, top color point, bottom color point)");
        }

        m_enabled = true;

        return model;
    }
};
