# Time Series Forecasting Python\* Demo

This README describes the Time Series Forecasting demo application.

## How It Works

Upon the start-up the demo application reads command line parameters and loads a network to Inference Engine.

The program provides an interactive CLI interface that gets a test dataset in .pickle format as input and draws predicted quantiles and ground truth curves.

## Supported datasets

* [Electricity](https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip).

### Prepare dataset

Our demo works with the test dataset in the .pickle format provided by accuracy_checker.

* Install accuracy_checker following [instruction](https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/README.md).
* Convert test dataset:
```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip
unzip LD2011_2014.txt.zip
convert_annotation electricity --data_path_file LD2011_2014.txt
```

### Running the Demo

Running the application with the `-h` option yields the following usage message:

```
usage: time_series_forecasting_demo.py [-h] -m MODEL [--input-name INPUT_NAME] [--output-name OUTPUT_NAME] [--data-path DATA_PATH] [--quantiles QUANTILES]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model
  --input-name INPUT_NAME
                        Optional. Name of the models input node.
  --output-name OUTPUT_NAME
                        Optional. Name of the models output node.
  --data-path DATA_PATH
                        Required. Path to the dataset file in .pickle format.
  --quantiles QUANTILES
                        Optional. Names of predicted quantiles.
```

To run the demo, you can use Intel's pretrained model. To download pretrained models, use the OpenVINO&trade; [Model Downloader](../../../tools/downloader/README.md). The list of models supported by the demo is in `<omz_dir>/demos/machine_translation_demo/python/models.lst`.

> **NOTE**: Before running the demo with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).


## Demo Output

The application draws predicted quantiles and ground truth curves.


## See Also

* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
