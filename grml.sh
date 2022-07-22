build_demo() {
    if [[ "$1" == "" ]]; then
        exit 1
    fi

    mkdir -p ${BINDIR} ${BUILDDIR}/omz_demos_build ${BUILDDIR}/intel64/Release
    docker run -it --rm \
        -e USER="$(id -u):$(id -g)" \
        -v "${ROOT}:/work:ro" \
        -v "${BUILDDIR}:/home/builder/omz_demos_build" \
        -v "${BINDIR}:/home/builder/omz_demos_build/intel64/Release" \
        wahtari/openvino_openmodelzoo_demos:builder \
        demos/build_demos.sh --target=$1

    docker build \
        --tag wahtari/openvino_openmodelzoo_demos:$1 \
        --file "${ROOT}/demos/$1/cpp_gapi/Dockerfile" \
        "${BINDIR}"
}

run_demo() {
    if [[ "$1" == "" ]]; then
        exit 1
    fi

    docker run -it --rm \
        -e USER="$(id -u):$(id -g)" \
        --privileged -v /dev:/dev \
        --net host \
        wahtari/openvino_openmodelzoo_demos:$1
}

build_napp() {
    if [[ "$1" == "" ]]; then
        exit 1
    fi

    mkdir -p ${NAPPDIR}
    docker save \
        --output "${NAPPDIR}/wahtari_openvino_openmodelzoo_demos_$1.docker.tar" \
        wahtari/openvino_openmodelzoo_demos:$1
}