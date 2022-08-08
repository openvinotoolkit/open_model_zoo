# Build OpenCV for OpenVINO

# Contents
- [Introduction](#introduction)
- [Building on Ubuntu](#building-on-ubuntu)
- [Building on Windows](#building-on-windows)

# Introduction
OpenVINO does not provide custom OpenCV drop since 2022.1.1 release.
If OpenVINO user needs OpenCV functionality there are 2 approaches how to get it:
1. Get OpenCV from another sources (system repositories, pip, conda, homebrew). It is easy to follow this approach, however it has several disadvantages:
   * OpenCV version is out-of-date  
   * OpenCV does not contain G-API module (e.g. some OMZ demos use G-API functionality)
   * OpenCV does not use available CPU instructions since it has build to cover wide range of hardware
   * OpenCV does not support Intel TBB, Intel Media SDK
   * OpenCV DNN module can not use OpenVINO as computational backend
2. Compile OpenCV from source code. This approach solves the issues mentioned above.

The instruction below shows how to build OpenCV for OpenVINO.

## Building on Ubuntu

### Prerequisites 
1. Install OpenVINO according to the [instruction](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html)
2. Install the following packages:
> sudo apt-get install \  
build-essential \  
cmake \  
ninja-build \  
libgtk-3-dev \  
libpng-dev \  
libjpeg-dev \  
libwebp-dev \  
libtiff5-dev \  
libopenexr-dev \  
libopenblas-dev \  
libx11-dev \  
libavutil-dev \  
libavcodec-dev \  
libavformat-dev \  
libswscale-dev \  
libavresample-devlibavutil-dev \  
libavcodec-dev \  
libavformat-dev \  
libswscale-dev \  
libavresample-dev \  
libtbb2 \  
libssl-dev \  
libva-dev \  
libmfx-dev \  
libgstreamer1.0-dev \  
libgstreamer-plugins-base1.0-dev 

### Procedure
1. Copy OpenCV repository:
> git clone --recurse-submodules https://github.com/opencv/opencv.git
2. Create build directory and enter into it:
> mkdir ~/build-opencv && cd ~/build-opencv
3. Compile and install OpenCV:
> cmake -G Ninja \  
-D BUILD_INFO_SKIP_EXTRA_MODULES=ON \  
-D BUILD_EXAMPLES=OFF \  
-D BUILD_JASPER=OFF \  
-D BUILD_JAVA=OFF \  
-D BUILD_JPEG=ON \  
-D BUILD_APPS_LIST=version \  
-D BUILD_opencv_apps=ON \  
-D BUILD_opencv_java=OFF \  
-D BUILD_OPENEXR=OFF \  
-D BUILD_PNG=ON \  
-D BUILD_TBB=OFF \  
-D BUILD_WEBP=OFF \  
-D BUILD_ZLIB=ON \  
-D WITH_1394=OFF \  
-D WITH_CUDA=OFF \  
-D WITH_EIGEN=OFF \  
-D WITH_GPHOTO2=OFF \  
-D WITH_GSTREAMER=ON \  
-D OPENCV_GAPI_GSTREAMER=OFF \  
-D WITH_GTK_2_X=OFF \  
-D WITH_IPP=ON \  
-D WITH_JASPER=OFF \  
-D WITH_LAPACK=OFF \  
-D WITH_MATLAB=OFF \  
-D WITH_MFX=ON \  
-D WITH_OPENCLAMDBLAS=OFF \  
-D WITH_OPENCLAMDFFT=OFF \  
-D WITH_OPENEXR=OFF \  
-D WITH_OPENJPEG=OFF \  
-D WITH_QUIRC=OFF \  
-D WITH_TBB=OFF \  
-D WITH_TIFF=OFF \  
-D WITH_VTK=OFF \  
-D WITH_WEBP=OFF \  
-D CMAKE_USE_RELATIVE_PATHS=ON \  
-D CMAKE_SKIP_INSTALL_RPATH=ON \  
-D ENABLE_BUILD_HARDENING=ON \  
-D ENABLE_CONFIG_VERIFICATION=ON \  
-D ENABLE_PRECOMPILED_HEADERS=OFF \  
-D ENABLE_CXX11=ON \  
-D INSTALL_PDB=ON \  
-D INSTALL_TESTS=ON \  
-D INSTALL_C_EXAMPLES=ON \  
-D INSTALL_PYTHON_EXAMPLES=ON \  
-D CMAKE_INSTALL_PREFIX=install \  
-D OPENCV_SKIP_PKGCONFIG_GENERATION=ON \  
-D OPENCV_SKIP_PYTHON_LOADER=OFF \  
-D OPENCV_SKIP_CMAKE_ROOT_CONFIG=ON \  
-D OPENCV_GENERATE_SETUPVARS=OFF \  
-D OPENCV_BIN_INSTALL_PATH=bin \  
-D OPENCV_INCLUDE_INSTALL_PATH=include \  
-D OPENCV_LIB_INSTALL_PATH=lib \  
-D OPENCV_CONFIG_INSTALL_PATH=cmake \  
-D OPENCV_3P_LIB_INSTALL_PATH=3rdparty \  
-D OPENCV_SAMPLES_SRC_INSTALL_PATH=samples \  
-D OPENCV_DOC_INSTALL_PATH=doc \  
-D OPENCV_OTHER_INSTALL_PATH=etc \  
-D OPENCV_LICENSES_INSTALL_PATH=etc/licenses \  
-D OPENCV_INSTALL_FFMPEG_DOWNLOAD_SCRIPT=ON \  
-D BUILD_opencv_world=OFF \  
-D BUILD_opencv_python2=OFF \  
-D BUILD_opencv_python3=ON \  
-D PYTHON3_PACKAGES_PATH=install/python/python3 \  
-D PYTHON3_LIMITED_API=ON \  
-D HIGHGUI_PLUGIN_LIST=all \  
-D OPENCV_PYTHON_INSTALL_PATH=python \  
-D CPU_BASELINE=SSE4_2 \  
-D OPENCV_IPP_GAUSSIAN_BLUR=ON \  
-D WITH_INF_ENGINE=ON \  
-D InferenceEngine_DIR=<OpenVINO_ROOT_DIRECTORY>/runtime/cmake/ \  
-D ngraph_DIR=<OpenVINO_ROOT_DIRECTORY>/runtime/cmake/ \  
-D INF_ENGINE_RELEASE=2022010000 \  
-D VIDEOIO_PLUGIN_LIST=ffmpeg,gstreamer,mfx \  
-D CMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \  
-D CMAKE_BUILD_TYPE=Release <OpenCV_ROOT_REPO_DIRECTORY> && \  
ninja && cmake --install .

OpenCV package is available at `~/build-opencv/install` directory. 

To compile application that uses OpenCV, the following environment variables should be specified:
> export OpenCV_DIR="<OpenCV_INSTALL_DIR>/cmake"  
export LD_LIBRARY_PATH="<OpenCV_INSTALL_DIR>/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  
export PYTHONPATH="<OpenCV_INSTALL_DIR>/python${PYTHONPATH:+:$PYTHONPATH}"


## Building on Windows

### Prerequisites 
1. Install Microsoft Visual Studio 
2. Install [cmake](https://cmake.org/download/)
1. Install OpenVINO according to the [instruction](https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_windows.html)
2. Install [Intel® Media SDK for Windows](https://www.intel.com/content/www/us/en/developer/tools/media-sdk/choose-download-client.html)

### Procedure
1. Copy OpenCV repository:
> git clone --recurse-submodules https://github.com/opencv/opencv.git
2. Create build directory and enter into it:
> mkdir "build-opencv" && cd "build-opencv"
3. Setup MSVC environment by running `vcvars64.bat`
4. Compile and install OpenCV:
> cmake -G Ninja ^  
-DBUILD_INFO_SKIP_EXTRA_MODULES=ON ^  
-DBUILD_EXAMPLES=OFF ^  
-DBUILD_JASPER=OFF ^  
-DBUILD_JAVA=OFF ^  
-DBUILD_JPEG=ON ^  
-DBUILD_APPS_LIST=version ^  
-DBUILD_opencv_apps=ON ^  
-DBUILD_opencv_java=OFF ^  
-DBUILD_OPENEXR=OFF ^  
-DBUILD_PNG=ON ^  
-DBUILD_TBB=OFF ^  
-DBUILD_WEBP=OFF ^  
-DBUILD_ZLIB=ON ^  
-DWITH_1394=OFF ^  
-DWITH_CUDA=OFF ^  
-DWITH_EIGEN=OFF ^  
-DWITH_GPHOTO2=OFF ^  
-DWITH_GSTREAMER=OFF ^  
-DOPENCV_GAPI_GSTREAMER=OFF ^  
-DWITH_GTK_2_X=OFF ^  
-DWITH_IPP=ON ^  
-DWITH_JASPER=OFF ^  
-DWITH_LAPACK=OFF ^  
-DWITH_MATLAB=OFF ^  
-DWITH_MFX=ON ^  
-DWITH_OPENCLAMDBLAS=OFF ^  
-DWITH_OPENCLAMDFFT=OFF ^  
-DWITH_OPENEXR=OFF ^  
-DWITH_OPENJPEG=OFF ^  
-DWITH_QUIRC=OFF ^  
-DWITH_TBB=OFF ^  
-DWITH_TIFF=OFF ^  
-DWITH_VTK=OFF ^  
-DWITH_WEBP=OFF ^  
-DCMAKE_USE_RELATIVE_PATHS=ON ^  
-DCMAKE_SKIP_INSTALL_RPATH=ON ^  
-DENABLE_BUILD_HARDENING=ON ^  
-DENABLE_CONFIG_VERIFICATION=ON ^  
-DENABLE_PRECOMPILED_HEADERS=OFF ^  
-DENABLE_CXX11=ON ^  
-DINSTALL_PDB=ON ^  
-DINSTALL_TESTS=ON ^  
-DINSTALL_C_EXAMPLES=ON ^  
-DINSTALL_PYTHON_EXAMPLES=ON ^  
-DCMAKE_INSTALL_PREFIX=install ^  
-DOPENCV_SKIP_PKGCONFIG_GENERATION=ON ^  
-DOPENCV_SKIP_PYTHON_LOADER=OFF ^  
-DOPENCV_SKIP_CMAKE_ROOT_CONFIG=ON ^  
-DOPENCV_GENERATE_SETUPVARS=OFF ^  
-DOPENCV_BIN_INSTALL_PATH=bin ^  
-DOPENCV_INCLUDE_INSTALL_PATH=include ^  
-DOPENCV_LIB_INSTALL_PATH=lib ^  
-DOPENCV_CONFIG_INSTALL_PATH=cmake ^  
-DOPENCV_3P_LIB_INSTALL_PATH=3rdparty ^  
-DOPENCV_SAMPLES_SRC_INSTALL_PATH=samples ^  
-DOPENCV_DOC_INSTALL_PATH=doc ^  
-DOPENCV_OTHER_INSTALL_PATH=etc ^  
-DOPENCV_LICENSES_INSTALL_PATH=etc/licenses ^  
-DOPENCV_INSTALL_FFMPEG_DOWNLOAD_SCRIPT=ON ^  
-DBUILD_opencv_world=OFF ^  
-DBUILD_opencv_python2=OFF ^  
-DBUILD_opencv_python3=ON ^  
-DPYTHON3_PACKAGES_PATH=install/python/python3 ^  
-DPYTHON3_LIMITED_API=ON ^  
-DOPENCV_PYTHON_INSTALL_PATH=python ^  
-DCPU_BASELINE=SSE4_2 ^  
-DOPENCV_IPP_GAUSSIAN_BLUR=ON ^  
-DWITH_INF_ENGINE=ON ^  
-DInferenceEngine_DIR="<OpenVINO_ROOT_DIRECTORY>\runtime\cmake" ^  
-Dngraph_DIR="<OpenVINO_ROOT_DIRECTORY>\runtime\cmake" ^  
-DINF_ENGINE_RELEASE=2022010000 ^  
-DVIDEOIO_PLUGIN_LIST=mfx,msmf ^  
-DCMAKE_BUILD_TYPE=Release <OpenCV_ROOT_REPO_DIRECTORY> &&  
ninja &&  
cmake --install .

OpenCV package is available at `build-opencv/install` directory. 

To compile application that uses OpenCV, the following environment variables should be specified:
>set "OpenCV_DIR=<OpenCV_INSTALL_DIR>\cmake"  
set "PATH=<OpenCV_INSTALL_DIR>\bin;%PATH%"  
set "PYTHONPATH=<OpenCV_INSTALL_DIR>\python;%PYTHONPATH%"  