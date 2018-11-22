Sample
===========

In this sample we will go through typical steps required to evaluate DL topologies. 

We will try to evaluate **SampLeNet** topology as an example.

### 1. Extract dataset

In this sample we will use toy dataset which we refer to as *sample dataset*, which contains 10k images 
of 10 different classes (classification problem), which is actually CIFAR10 dataset converted to png.

```bash
tar xvf sample/sample_dataset.tar.gz -C sample
```

### 2. Convert annotation 

Dataset annotation should be converted to a common annotation representation format.
In order to do this you need to provide your own **annotation converter**, 
i.e. implement `BaseFormatConverter` interface. 
All annotation converters are stored in `annotation_converters` directory. 

There is already annotation converter for *sample dataset*, so you do not need to write your own to execute this sample. 
Study its code in `annotation_converters/sample_converter.py` if you need to write annotation converter for your topology

Convert annotation: 
```bash
mkdir sample/annotation
python convert_annotation.py sample sample/sample_dataset -o sample/annotation -a sample_annotation 
```

First command line option `sample` is a name for annotation converter; second command line option `sample/sample_dataset` is a path to extracted *sample dataset*.
`-o` specifies destination directory for converted annotation; `-a` specifies file name for converted annotation


### 3. Evaluate sample topology

Typically you need to write configuration file, describing evaluation process of your topology. 
There is already config file for evaluating SampLeNet using Caffe framework, read it carefully.

```bash
accuracy_check -c sample/sample_config.yml -m data/test_models -s sample -a sample/annotation
```

Used options: `-c` path to evaluation config, `-m` directory where models are stored, `-s` directory where source data (datasets),
`-a` directory with converted annotation files.

If everything worked correctly, you should be able to get `75.02%` accuracy. 

Now try edit config, to run SampLeNet on Inference Engine, or go directly to your topology!
