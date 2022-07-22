FROM wahtari/openvino_openmodelzoo_demos:runtime

RUN mkdir /models && cd /models && \
    wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml && \
    wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/human-pose-estimation-0001/FP32/human-pose-estimation-0001.bin

COPY human_pose_estimation_demo /demo

CMD ["/demo", "-no-show", "-at", "openpose", "-i", "/models/pedestrian-detection-adas-0002.bin", "-m", "/models/pedestrian-detection-adas-0002.xml"]