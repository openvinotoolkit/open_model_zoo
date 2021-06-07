/*********************************************************************
* Copyright (c) 2020 Intel Corporation
* SPDX-License-Identifier: Apache-2.0
**********************************************************************/

#ifndef YOKLM_SORTED_SEARCH_HPP
#define YOKLM_SORTED_SEARCH_HPP

#include <limits>
#include <stdexcept>


namespace yoklm {

// Given a sorted (non-decreasing) array, find index m in range [l, r), such that value = array[m].
// If value is absent, return not_found.
//   Preconditions:
// IndexT must fit into double (52 bits)
// 0 <= l <= r < (1<<52)  (non-negativity for proper rounding)
// plv = array[l-1] or ValueT range minimum (or at least plv <= array[l])
// rv = array[r] or ValueT range maximum (or at least rv >= array[r-1])
template <typename ArrayT, typename IndexT, typename ValueT>
IndexT secant_search(const ArrayT& array, IndexT l, IndexT r, ValueT plv, ValueT rv, const ValueT not_found, const ValueT value) {
  // We'll maintain values in the form (array[i] - value)
  double plv_ = plv - (double)value;
  double rv_ = rv - (double)value;

  while (l + 1 < r) {
    // (r-l+1) is not a bug. Because expectation of k-th value (k=1...n) in sorted array of n i.i.d. random samples from U(lv,rv) is lv+(rv-lv)*k/(n+1).
    double m_ = l + (r - l + 1) * (-plv_ / (rv_ - plv_)) - 0.5;
    if (m_ < l)  m_ = l;
    if (m_ > r-1)  m_ = r-1;
    IndexT m = m_;

    const double mv_ = array[m] - (double)value;
    if (mv_ < 0) {
      l = m + 1;
      plv_ = mv_;
    } else if (mv_ > 0) {
      r = m;
      rv_ = mv_;
    } else {
      return m;
    }
  }

  if (l < r && array[l] == value)
    return l;
  else
    return not_found;
}

// Find the last (rightmost) m0 in [l,r), so that array[m0] <= value.
//   Preconditions:
// Array is non-decreasing
// the actual value in array[l] is ignored (never read), and the algo works as if array[l] <= value
template <typename ArrayT, typename IndexT, typename ValueT>
IndexT binary_search(const ArrayT& array, IndexT l, IndexT r, const ValueT value) {
  if (l >= std::numeric_limits<IndexT>::max() - r)
    throw std::runtime_error("yoklm::binary_search: index is too large.  Broken LM file?");

  while (l + 1 < r) {
    // Invariant: m0 (the answer) is in [l,r), that is
    //   array[l] <= value and array[r] > value (-inf beyond left, +inf beyond right)
    IndexT m = (l + r) / 2;  // m is in (l,r), so both (r-m) and (m-l) are less than (r-l)
    if (array[m] <= value)
      l = m;
    else
      r = m;
  }

  return l;
}

} // namespace yoklm


#endif // YOKLM_SORTED_SEARCH_HPP
