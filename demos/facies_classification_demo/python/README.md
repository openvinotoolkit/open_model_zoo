# Facies classification Python* Demo

This demo demonstrate how to run facies classification using OpenVINO&trade;

This model came from seismic interpretation tasks. Fasies is the overall characteristics of a rock unit that reflect its origin and differentiate the unit from others around it.  Mineralogy and sedimentary source, fossil content, sedimentary structures and texture distinguish one facies from another. Data are presented in the 3D arrays.


## How It Works
Upon the start-up, the demo application loads a network and an given dataset file to the Inference Engine plugin. When inference is done, the application displays 3d itkwidget viewer with facies interpretation.

## Running

### Installation of dependencies

Steps to create CPU extension:

```bash
source <openvino_install>/bin/setupvars.sh
export TBB_DIR=${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/tbb/cmake/

cd ../user_ie_extensions
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc --all)
mv libunpool_cpu_extension.so ../../python
```

To install required dependencies run

```bash
$ python -m pip install -r requirements.txt
```


To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [xxx] <- (link)

Run Jupyter notebook with demo
```bash
$ jupyter notebook
```

## Demo Output

The application uses Jupyter notebook to display 3d itkwidget with resulting instance classification masks.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
