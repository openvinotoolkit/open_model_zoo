# Caffe Installation Tips

## Install OpenCV 3.3 or later with Python3 bindings

Accuracy Checker uses OpenCV library for image processing. You can miss this step if you are using OpenCV from [OpenVINO toolkit][openvino-get-started].

```bash
sudo apt-get install libopencv-dev
python3 -m pip install opencv-python
```

If you need to use custom OpenCV version, you can find installation instructions [here][opencv-install].

## Install Caffe with Python3 bindings

* Clone repository:

```bash
git clone https://github.com/BVLC/caffe.git
cd caffe
```

* Install Caffe dependencies:

```bash
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install --no-install-recommends libboost-all-dev
python3 -m pip install --user -r python/requirements.txt
python3 -m pip install --user matplotlib
```

* Build

If you need CPU only version of caffe add `-DCPU_ONLY=ON` to cmake command.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<caffe/install/dir> -Dpython_version=3 -DBLAS=open ..
make
sudo make install
```

* Copy Python library to your python installation.

  * If you are NOT using virtual env:

    ```bash
    cp -r ../python/caffe $(python3 -m site --user-site)
    cp --remove-destination lib/_caffe.so $(python3 -m site --user-site)/caffe
    ```

  * If you are using virtual env:

    ```bash
    cp -r ../python/caffe $VIRTUAL_ENV/lib/python3.5/site-packages
    cp --remove-destination lib/_caffe.so $VIRTUAL_ENV/lib/python3.5/site-packages/caffe
    ```

## Check your installation

You can test prerequisites with the following command. If it does not fail, then you are installed prerequisites correctly:

```bash
python3 -c 'import caffe, cv2'
```

[openvino-get-started]: https://software.intel.com/en-us/openvino-toolkit/documentation/get-started
[opencv-install]: https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html
