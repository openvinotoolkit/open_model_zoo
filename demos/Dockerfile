FROM docker.wahtari.m/nlab/builder:v0.0.28

# Install OpenVINO and OpenCV.
RUN export DEBIAN_FRONTEND=noninteractive && \
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    echo "deb https://apt.repos.intel.com/openvino/2022 focal main" | tee /etc/apt/sources.list.d/intel-openvino-2022.list && \
    apt -y update && \
    apt-cache search openvino && \
    apt -y install --no-install-recommends \
        git \
        openvino \
        openvino-opencv && \
    apt -y clean
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino_2022

# Install Wahtari controller-libs.
RUN cd /tmp && \
    git clone https://github.com/wahtari/controller-libs.git && \
    cd controller-libs/cpp && \
    mkdir /usr/local/include/controller-libs && \
    mv \
        libnlab-ctrl.h \
        libnlab-ctrl.hpp \
        /usr/local/include/controller-libs && \
    mv \
        libnlab-ctrl.so \
        libnlab-ctrl-cpp.so \
        /usr/local/lib/ && \
    cd /tmp && \
    rm -rf /tmp/controller-libs
 