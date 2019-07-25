Sample
===========

In this sample we will go through typical steps required to evaluate DL topologies. 

We will try to evaluate **SampLeNet** topology as an example.

### 1. Download and extract dataset

In this sample we will use toy dataset which we refer to as *sample dataset*, which contains 10k images 
of 10 different classes (classification problem), which is actually CIFAR10 dataset converted to png (image conversion will be done automatically in evaluation process)

You can download original CIFAR10 dataset from [official website][cifar_python_download].

Extract downloaded dataset to sample directory


```bash
tar xvf cifar-10-python.tar.gz -C sample
```

### 2. Evaluate sample topology

Typically you need to write configuration file, describing evaluation process of your topology.
There is already config file for evaluating SampLeNet using OpenVINO framework, read it carefully. It runs Caffe model using Model Optimizer which requires installed Caffe. If you have not opportunity to use Caffe, please replace `caffe_model` and `caffe_weights` on

```yaml
model: SampleNet.xml
weights: SampleNet.bin
```

```bash
accuracy_check -c sample/sample_config.yml -m data/test_models -s sample
```

Used options: `-c` path to evaluation config, `-m` directory where models are stored, `-s` directory where source data (datasets).

If everything worked correctly, you should be able to get `75.02%` accuracy.

Now try edit config, to run SampLeNet on other device or framework (e.g. Caffe, MxNet or OpenCV), or go directly to your topology!

[cifar_python_download]: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
