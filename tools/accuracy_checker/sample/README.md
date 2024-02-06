# Sample

In this sample we will go through typical steps required to evaluate DL topologies.

We will try to evaluate **SampLeNet** topology as an example.

### 1. Download and extract dataset

In this sample we will use toy dataset which we refer to as *sample dataset*, which contains 10K images
of 10 different classes (classification problem), which is actually CIFAR10 dataset converted to PNG (image conversion will be done automatically in evaluation process)

You can download original CIFAR10 dataset from [official website](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

Extract downloaded dataset to sample directory


```bash
tar xvf cifar-10-python.tar.gz -C sample
```

### 2. Evaluate sample topology

Typically you need to write a configuration file describing evaluation process of your topology.
There is already a prepared config file for evaluating SampLeNet using the OpenVINO framework at `<omz_dir>/tools/accuracy_checker/sample/sample_config.yml`, so please read it carefully before using.

Then run Accuracy Checker with the following command:

```bash
accuracy_check -c sample/sample_config.yml -m data/test_models -s sample
```

Used options: `-c` path to evaluation config, `-m` directory where models are stored, `-s` directory where source data (datasets).

If everything worked correctly, you should be able to get `75.02%` accuracy.

Now try edit config, to run SampLeNet on other device or framework (e.g., Caffe, MXNet or OpenCV), or go directly to your topology!

###  Additional useful resources

* large collection of configuration files for models from Open Model Zoo. You can find instruction how to use predefined configuration [here](../configs/README.md)
* config at `<omz_dir>/tools/accuracy_checker/sample/opencv_sample_config.yml` for running SampleNet via [OpenCV launcher](../accuracy_checker/launcher/opencv_launcher_readme.md).
