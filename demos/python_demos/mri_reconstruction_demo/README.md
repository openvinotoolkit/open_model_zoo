# Magnetic Resonance Image Reconstruction Python Demo

This demo demonstrates MRI reconstruction model described in https://arxiv.org/abs/1810.12473 and implemented in https://github.com/rmsouza01/Hybrid-CS-Model-MRI/.
The model is used to restore undersampled MRI scans which is useful for data compression.


## Running

1. Once you need to build extensions library with FFT implementation.
    ```bash
    cd open_model_zoo/demos
    mkdir build && cd build

    source /opt/intel/openvino/bin/setupvars.sh
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make --jobs=$(nproc) fft_cpu_extension
    ```

2. Running the application with the -h option yields the following usage message:
    ```bash
    $ python3 mri_reconstruction_demo.py -h

    usage: mri_reconstruction_demo.py [-h] [-i INPUT] [-m MODEL]
                                      [-l CPU_EXTENSION] [-d DEVICE] [-p PATTERN]

    MRI reconstrution demo for network from https://github.com/rmsouza01/Hybrid-
    CS-Model-MRI (https://arxiv.org/abs/1810.12473)

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Path to input .npy file with MRI scan data.
      -m MODEL, --model MODEL
                            Path to .xml file of OpenVINO IR.
      -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                            Path to extensions library with FFT implementation.
      -d DEVICE, --device DEVICE
                            Optional. Specify the target device to infer on; CPU,
                            GPU, HDDL or MYRIAD is acceptable. For non-CPU
                            targets, HETERO plugin is used with CPU fallbacks to
                            FFT implementation. Default value is CPU
      -p PATTERN            Path to sampling mask in .npy format.
    ```

3. To run the demo, you need to have
  * A sample scan from [Calgary-Campinas Public Brain MR Dataset](https://sites.google.com/view/calgary-campinas-dataset/home)
  * Trained network in OpenVINO IR format (follow [Convert model](#convert_model) chapter)
  * [Sampling mask](https://github.com/rmsouza01/Hybrid-CS-Model-MRI/blob/master/Data/sampling_mask_20perc.npy)

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

3. Copy extensions to Model Optimizer (`FFT2D`, `IFFT2D`, `Complex`, `ComplexAbs`):
    ```bash
    cp -r python_demos/mri_reconstruction_demo/mo_extensions/* /opt/intel/openvino/deployment_tools/model_optimizer/extensions/
    ```
4. Generate OpenVINO IR:
    ```bash
    python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
        --input_model wnet_20.pb \
        --input_shape "[1, 256, 256, 2]"
    ```
