ie_add_cpu_extension(NAME fft_cpu_extension
                     SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ext_fft2d.cpp"
                     DEPENDENCIES opencv_core)

ie_add_sample(NAME mri_reconstruction_demo
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/npy_reader.hpp"
              DEPENDENCIES fft_cpu_extension
              OPENCV_DEPENDENCIES core highgui)
