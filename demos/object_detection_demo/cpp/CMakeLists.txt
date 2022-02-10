# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_demo(NAME object_detection_demo
    SOURCES main.cpp
    DEPENDENCIES monitors models pipelines)

add_demo(NAME prop
    SOURCES proposal.cpp
    HEADERS proposal_common.hpp
    DEPENDENCIES monitors models pipelines)  # TODO: remove pipelines
