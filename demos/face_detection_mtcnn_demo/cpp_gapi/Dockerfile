FROM wahtari/openvino_openmodelzoo_demos:runtime

RUN mkdir /models && cd /models && \
    wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-0200/FP32/face-detection-0200.xml && \
    wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/face-detection-0200/FP32/face-detection-0200.bin

COPY face_detection_mtcnn_demo /demo

CMD ["/demo", "-no-show", "-at", "openpose", "-i", "/models/face-detection-0200.bin", "-m", "/models/face-detection-0200.xml"]