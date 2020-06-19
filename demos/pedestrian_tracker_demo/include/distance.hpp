// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <memory>
#include <vector>

#include <inference_engine.hpp>

///
/// \brief The IDescriptorDistance class declares an interface for distance
/// computation between reidentification descriptors.
///
class IDescriptorDistance {
public:
    ///
    /// \brief Computes distance between two descriptors.
    /// \param[in] descr1 First descriptor.
    /// \param[in] descr2 Second descriptor.
    /// \return Distance between two descriptors.
    ///
    virtual float Compute(const cv::Mat &descr1, const cv::Mat &descr2) = 0;

    ///
    /// \brief Computes distances between two descriptors in batches.
    /// \param[in] descrs1 Batch of first descriptors.
    /// \param[in] descrs2 Batch of second descriptors.
    /// \return Distances between descriptors.
    ///
    virtual std::vector<float> Compute(const std::vector<cv::Mat> &descrs1,
                                       const std::vector<cv::Mat> &descrs2) = 0;

    virtual ~IDescriptorDistance() {}
};

///
/// \brief The CosDistance class allows computing cosine distance between two
/// reidentification descriptors.
///
class CosDistance : public IDescriptorDistance {
public:
    ///
    /// \brief CosDistance constructor.
    /// \param[in] descriptor_size Descriptor size.
    ///
    explicit CosDistance(const cv::Size &descriptor_size);

    ///
    /// \brief Computes distance between two descriptors.
    /// \param descr1 First descriptor.
    /// \param descr2 Second descriptor.
    /// \return Distance between two descriptors.
    ///
    float Compute(const cv::Mat &descr1, const cv::Mat &descr2) override;

    ///
    /// \brief Computes distances between two descriptors in batches.
    /// \param[in] descrs1 Batch of first descriptors.
    /// \param[in] descrs2 Batch of second descriptors.
    /// \return Distances between descriptors.
    ///
    std::vector<float> Compute(
        const std::vector<cv::Mat> &descrs1,
        const std::vector<cv::Mat> &descrs2) override;

private:
    cv::Size descriptor_size_;
};



///
/// \brief Computes distance between images
///        using MatchTemplate function from OpenCV library
///        and its cross-correlation computation method in particular.
///
class MatchTemplateDistance : public IDescriptorDistance {
public:
    ///
    /// \brief Constructs the distance object.
    ///
    /// \param[in] type Method of MatchTemplate function computation.
    /// \param[in] scale Scale parameter for the distance.
    ///            Final distance is computed as:
    ///            scale * distance + offset.
    /// \param[in] offset Offset parameter for the distance.
    ///            Final distance is computed as:
    ///            scale * distance + offset.
    ///
    MatchTemplateDistance(int type = cv::TemplateMatchModes::TM_CCORR_NORMED,
                          float scale = -1, float offset = 1)
        : type_(type), scale_(scale), offset_(offset) {}
    ///
    /// \brief Computes distance between image descriptors.
    /// \param[in] descr1 First image descriptor.
    /// \param[in] descr2 Second image descriptor.
    /// \return Distance between image descriptors.
    ///
    float Compute(const cv::Mat &descr1, const cv::Mat &descr2) override;
    ///
    /// \brief Computes distances between two descriptors in batches.
    /// \param[in] descrs1 Batch of first descriptors.
    /// \param[in] descrs2 Batch of second descriptors.
    /// \return Distances between descriptors.
    ///
    std::vector<float> Compute(const std::vector<cv::Mat> &descrs1,
                               const std::vector<cv::Mat> &descrs2) override;
    virtual ~MatchTemplateDistance() {}

private:
    int type_;      ///< Method of MatchTemplate function computation.
    float scale_;   ///< Scale parameter for the distance. Final distance is
                    /// computed as: scale * distance + offset.
    float offset_;  ///< Offset parameter for the distance. Final distance is
                    /// computed as: scale * distance + offset.
};

