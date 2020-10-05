# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

FILE(GLOB SRC_FILES ./*.cpp ../common/processing/*.cpp)
FILE(GLOB H_FILES ./*.h ../common/processing/*.h)

ie_add_sample(NAME async_pipeline_demo
              SOURCES ${SRC_FILES} ../segmentation_demo_async/segmentation_pipeline.cpp ../object_detection_demo_ssd_async/detection_pipeline.cpp ../object_detection_demo_ssd_async/detection_pipeline_ssd.cpp ../object_detection_demo_ssd_async/detection_pipeline_yolo.cpp
              HEADERS ${H_FILES}
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui videoio imgproc)

target_include_directories(async_pipeline_demo PRIVATE ../common/processing/)
target_include_directories(async_pipeline_demo PRIVATE ../object_detection_demo_ssd_async/)
target_include_directories(async_pipeline_demo PRIVATE ../segmentation_demo_async/)
target_link_libraries(async_pipeline_demo PRIVATE ngraph::ngraph)
