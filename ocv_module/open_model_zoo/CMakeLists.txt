set(the_description "Open Model Zoo")
ocv_add_module(open_model_zoo opencv_core WRAP python)

set(omz_topologies_hdr "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2/open_model_zoo/topologies.hpp")
set(omz_topologies_impl "${CMAKE_CURRENT_SOURCE_DIR}/misc/python/pyopencv_open_model_zoo_topologies.hpp")

execute_process(COMMAND "${PYTHON_DEFAULT_EXECUTABLE}"
                        "${CMAKE_CURRENT_SOURCE_DIR}/gen.py"
                        "${CMAKE_CURRENT_SOURCE_DIR}/../../tools/downloader/list_topologies.yml"
                        ${omz_topologies_hdr} ${omz_topologies_impl})

ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()
