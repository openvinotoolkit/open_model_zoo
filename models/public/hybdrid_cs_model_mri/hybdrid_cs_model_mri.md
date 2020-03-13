# Hybrid-CS-Model-MRI

## Use Case and High-Level Description

Undersampled MRI reconstruction model described in https://arxiv.org/pdf/1810.12473.pdf and implemented in https://github.com/rmsouza01/Hybrid-CS-Model-MRI.

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Image processing                          |
| GFlops                          | 00.000                                    |
| MParams                         | 00.000                                    |
| Source framework                | TensorFlow\*                              |

## Performance

On https://sites.google.com/view/calgary-campinas-dataset/home validation dataset it is reported to achieve `35.772 (mean) +/- 3.214 (std) Db` (PSNR).

## Convert model

1. Download repository https://github.com/rmsouza01/Hybrid-CS-Model-MRI
    ```bash
    git clone https://github.com/rmsouza01/Hybrid-CS-Model-MRI
    ```

2. Convert pre-trained `.hdf5` to frozen `.pb` graph (tested with TensorFlow==1.15.0 and Keras==2.2.4):
    ```python
    import numpy as np
    import frequency_spatial_network as fsnet

    under_rate = '20'

    stats = np.load("Hybrid-CS-Model-MRI/Data/stats_fs_unet_norm_" + under_rate + ".npy")
    var_sampling_mask = np.load("Hybrid-CS-Model-MRI/Data/sampling_mask_" + under_rate + "perc.npy")

    model = fsnet.wnet(stats[0], stats[1], stats[2], stats[3], kshape = (5,5), kshape2=(3,3))

    model_name = "Hybrid-CS-Model-MRI/Models/wnet_" + under_rate + ".hdf5"
    model.load_weights(model_name)

    inp = np.random.standard_normal([1, 256, 256, 2]).astype(np.float32)
    np.save('inp', inp)

    import keras as K

    sess = K.backend.get_session()
    sess.as_default()

    graph_def = sess.graph.as_graph_def()
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['conv2d_44/BiasAdd'])

    with tf.gfile.FastGFile('wnet_20.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())
    ```

3. Generate OpenVINO IR:
    ```bash
    python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
        --input_model wnet_20.pb \
        --input_shape "[1, 256, 256, 2]" \
        --extensions /path/to/open_model_zoo/models/public/hybdrid_cs_model_mri/mo_extensions
    ```

## Legal Information

The original model is distributed under the
[MIT](https://raw.githubusercontent.com/rmsouza01/Hybrid-CS-Model-MRI/master/LICENSE).
