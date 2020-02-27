// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <ie_plugin_ptr.hpp>
#include <inference_engine.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"


#include "core.hpp"
#include "cnn.hpp"

///
/// \brief The IImageDescriptor class declares base class for image
/// descriptor.
///
class IImageDescriptor {
public:
    ///
    /// \brief Descriptor size getter.
    /// \return Descriptor size.
    ///
    virtual cv::Size size() const = 0;

    ///
    /// \brief Computes image descriptor.
    /// \param[in] mat Color image.
    /// \param[out] descr Computed descriptor.
    ///
    virtual void Compute(const cv::Mat &mat, cv::Mat *descr) = 0;

    ///
    /// \brief Computes image descriptors in batches.
    /// \param[in] mats Images of interest.
    /// \param[out] descrs Matrices to store the computed descriptors.
    ///
    virtual void Compute(const std::vector<cv::Mat> &mats,
                         std::vector<cv::Mat> *descrs) = 0;

    ///
    /// \brief Prints performance counts for CNN-based descriptors
    ///
    virtual void PrintPerformanceCounts(std::string fullDeviceName) const {}

    virtual ~IImageDescriptor() {}
};


///
/// \brief Uses resized image as descriptor.
///
class ResizedImageDescriptor : public IImageDescriptor {
public:
    ///
    /// \brief Constructor.
    /// \param[in] descr_size Size of the descriptor (resized image).
    /// \param[in] interpolation Interpolation algorithm.
    ///
    explicit ResizedImageDescriptor(const cv::Size &descr_size,
                                    const cv::InterpolationFlags interpolation)
        : descr_size_(descr_size), interpolation_(interpolation) {
            PT_CHECK_GT(descr_size.width, 0);
            PT_CHECK_GT(descr_size.height, 0);
        }

    ///
    /// \brief Returns descriptor size.
    /// \return Number of elements in the descriptor.
    ///
    cv::Size size() const override { return descr_size_; }

    ///
    /// \brief Computes image descriptor.
    /// \param[in] mat Frame containing the image of interest.
    /// \param[out] descr Matrix to store the computed descriptor.
    ///
    void Compute(const cv::Mat &mat, cv::Mat *descr) override {
        PT_CHECK(descr != nullptr);
        PT_CHECK(!mat.empty());
        cv::resize(mat, *descr, descr_size_, 0, 0, interpolation_);
    }

    ///
    /// \brief Computes images descriptors.
    /// \param[in] mats Frames containing images of interest.
    /// \param[out] descrs Matrices to store the computed descriptors.
    //
    void Compute(const std::vector<cv::Mat> &mats,
                 std::vector<cv::Mat> *descrs) override  {
        PT_CHECK(descrs != nullptr);
        descrs->resize(mats.size());
        for (size_t i = 0; i < mats.size(); i++)  {
            Compute(mats[i], &(descrs[i]));
        }
    }

private:
    cv::Size descr_size_;

    cv::InterpolationFlags interpolation_;
};


class DescriptorIE : public IImageDescriptor {
private:
    VectorCNN handler;

public:
    DescriptorIE(const CnnConfig& config,
                 const InferenceEngine::Core& ie,
                 const std::string & deviceName):
        handler(config, ie, deviceName) {}

    ///
    /// \brief Descriptor size getter.
    /// \return Descriptor size.
    ///
    virtual cv::Size size() const {
        return cv::Size(1, handler.size());
    }

    ///
    /// \brief Computes image descriptor.
    /// \param[in] mat Color image.
    /// \param[out] descr Computed descriptor.
    ///
    virtual void Compute(const cv::Mat &mat, cv::Mat *descr) {
        handler.Compute(mat, descr);
    }

    ///
    /// \brief Computes image descriptors in batches.
    /// \param[in] mats Images of interest.
    /// \param[out] descrs Matrices to store the computed descriptors.
    ///
    virtual void Compute(const std::vector<cv::Mat> &mats,
                         std::vector<cv::Mat> *descrs) {
        handler.Compute(mats, descrs);
    }

    virtual void PrintPerformanceCounts(std::string fullDeviceName) const {
        handler.PrintPerformanceCounts(fullDeviceName);
    }
};

