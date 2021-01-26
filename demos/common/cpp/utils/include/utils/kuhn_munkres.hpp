// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "opencv2/core.hpp"

#include <memory>
#include <vector>


///
/// \brief The KuhnMunkres class
///
/// Solves the assignment problem.
///
class KuhnMunkres {
public:
    ///
    /// \brief Initializes the class for assignment problem solving.
    /// \param[in] greedy If a faster greedy matching algorithm should be used.
    explicit KuhnMunkres(bool greedy = false);

    ///
    /// \brief Solves the assignment problem for given dissimilarity matrix.
    /// It returns a vector that where each element is a column index for
    /// corresponding row (e.g. result[0] stores optimal column index for very
    /// first row in the dissimilarity matrix).
    /// \param dissimilarity_matrix CV_32F dissimilarity matrix.
    /// \return Optimal column index for each row. -1 means that there is no
    /// column for row.
    ///
    std::vector<size_t> Solve(const cv::Mat &dissimilarity_matrix);

private:
    class Impl;
    std::shared_ptr<Impl> impl_;  ///< Class implementation.
};
