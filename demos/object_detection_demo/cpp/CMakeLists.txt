# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_demo(NAME object_detection_demo
    SOURCES proposal.cpp
    HEADERS proposal_common.hpp
    DEPENDENCIES monitors models pipelines)
