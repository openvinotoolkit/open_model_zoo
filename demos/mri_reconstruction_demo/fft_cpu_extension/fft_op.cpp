// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fft_op.hpp"

using namespace TemplateExtension;

constexpr ngraph::NodeTypeInfo FFTOp::type_info;

//! [op:ctor]
FFTOp::FFTOp(const ngraph::Output<ngraph::Node>& inp, bool _inverse) : Op({inp}) {
    constructor_validate_and_infer_types();
    inverse = _inverse;
}
//! [op:ctor]

//! [op:validate]
void FFTOp::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), outShape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ngraph::Node> FFTOp::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 1) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<FFTOp>(new_args.at(0), inverse);
}
//! [op:copy]

//! [op:visit_attributes]
bool FFTOp::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("inverse", inverse);
    return true;
}
//! [op:visit_attributes]
