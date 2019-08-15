set(the_description "Open Model Zoo")
ocv_add_module(open_model_zoo opencv_core OPTIONAL opencv_dnn WRAP python)

set(omz_topologies_hdr "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/open_model_zoo/topologies.hpp")
set(omz_topologies_py_impl "${CMAKE_CURRENT_SOURCE_DIR}/misc/python/pyopencv_open_model_zoo_topologies.hpp")
set(omz_topologies_cpp_impl "${CMAKE_CURRENT_SOURCE_DIR}/src/topologies.cpp")

execute_process(COMMAND "${PYTHON_DEFAULT_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/gen.py"
                        ${omz_topologies_hdr} ${omz_topologies_py_impl} ${omz_topologies_cpp_impl}
                        "${CMAKE_CURRENT_SOURCE_DIR}/../../models/intel"
                        "${CMAKE_CURRENT_SOURCE_DIR}/../../models/public")

ocv_glob_module_sources()
ocv_module_include_directories()

if(INF_ENGINE_TARGET)
  file(GLOB demo_srcs "${CMAKE_CURRENT_SOURCE_DIR}/../../demos/human_pose_estimation_demo/src/*")
  list(APPEND OPENCV_MODULE_${the_module}_SOURCES ${demo_srcs})

  file(GLOB demo_srcs "${CMAKE_CURRENT_SOURCE_DIR}/../../demos/text_detection_demo/src/*")
  ocv_list_filterout(demo_srcs "image_grabber*")
  list(APPEND OPENCV_MODULE_${the_module}_SOURCES ${demo_srcs})
endif()

if(NOT HAVE_opencv_dnn)
  ocv_list_filterout(OPENCV_MODULE_${the_module}_HEADERS "dnn.hpp")
endif()

ocv_create_module(${INF_ENGINE_TARGET})
ocv_add_accuracy_tests()

if(INF_ENGINE_TARGET)
  target_include_directories(${the_module}
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../demos/human_pose_estimation_demo/include/"
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../demos/text_detection_demo/include/"
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../demos/common"
  )
  ocv_warnings_disable(CMAKE_CXX_FLAGS -Wnon-virtual-dtor -Wshadow)
endif()
