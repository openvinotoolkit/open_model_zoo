# Facies classification Python* Demo

This demo demonstrate how to run facies classification using OpenVINO&trade;


## How It Works
Upon the start-up, the demo application loads a network and an given dataset file to the Inference Engine plugin. When inference is done, the application displays 3d itkwidget viewer with facies interpretation.

## Running

### Installation of dependencies

Steps to create CPU extension:

```bash
source /opt/intel/openvino/bin/setupvars.sh
export TBB_DIR=/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/cmake/

cd user_ie_extensions
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc --all)
mv libuser_cpu_extension.so ../../libuser_cpu_extension_for_max_unpool.so
```

To install required dependencies run

```bash
pip install -r requirements.txt
```


To run the demo, you can use public or pre-trained models. You can download the pre-trained models with the OpenVINO [Model Downloader](../../../tools/downloader/README.md) or from [xxx] <- (link)

Run jupyter notebook and select `facies_demo.ipynd`
```bash
$ jupyter notebook
```

## Demo Output

The application uses jupyter notebook to display 3d itkwidget with resulting instance classification masks.

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
