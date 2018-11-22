# How to configurate Caffe launcher

For enabling Caffe launcher you need to add `framework: caffe` in launchers section of your configuration file and provide following parameters:

* `device` - specifing which device will be used for infer (`cpu`, 'gpu_0' and so on).
* `model` - path to prototxt file with Caffe model for your topology.
* `weights` - path to caffemodel file with weights for your topology.
* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here][adapters].

You also can specify batch size for your model using `batch`.

Caffe launcher config example:

```yml
launchers:
  - framework: caffe
    device: CPU
    model: path_to_model/alexnet.prototxt
    weights: path_to_weights/alexnet.caffemodel
    adapter: classification
    batch: 4
```

[adapters]: accuracy_checker/adapter/README.md
