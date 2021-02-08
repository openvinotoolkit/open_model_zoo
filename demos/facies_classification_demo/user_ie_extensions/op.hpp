// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

//! [op:header]
namespace TemplateExtension {

class Operation : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"MaxPoolGrad", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    Operation() = default;
    Operation(const ngraph::Output<ngraph::Node>& poolInp,
              const ngraph::Output<ngraph::Node>& poolOut,
              const ngraph::Output<ngraph::Node>& inp,
              const ngraph::Output<ngraph::Node>& shape);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
};
//! [op:header]

}  // namespace TemplateExtension
