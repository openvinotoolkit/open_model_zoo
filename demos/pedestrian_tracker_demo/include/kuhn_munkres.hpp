// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core.hpp"

#include <memory>
#include <vector>


///
/// \brief The KuhnMunkres class
///
/// Solves the assignment problem.
///
class KuhnMunkres {
public:
    KuhnMunkres();

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
    static constexpr int kStar = 1;
    static constexpr int kPrime = 2;

    cv::Mat dm_;
    cv::Mat marked_;
    std::vector<cv::Point> points_;

    std::vector<int> is_row_visited_;
    std::vector<int> is_col_visited_;

    int n_;

    void TrySimpleCase();
    bool CheckIfOptimumIsFound();
    cv::Point FindUncoveredMinValPos();
    void UpdateDissimilarityMatrix(float val);
    int FindInRow(int row, int what);
    int FindInCol(int col, int what);
    void Run();
};

