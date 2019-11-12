Language Model Python* Demo
===============================

This is the demo application for Language model, which predict next word from previous input.
To download the model for IR conversion, please follow the instruction:
 - For UNIX*-like systems:  
    1.Create new directory to store the model:   
    ```
      mkdir lm_1b
    ```
    2.Go to the lm_1b directory:   
    ```
      cd lm_1b
    ```
    3.Download the model GraphDef file:   
    ```
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt  
    ```  
    4.Create new directory to store 12 checkpoint shared files:   
    ```
      mkdir ckpt
    ```  
    5.Go to the ckpt directory:   
    ```
      cd ckpt
    ```  
    6.Download 12 checkpoint shared files:  
    ```
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7  
      wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8  
    ```
 - For Windows*-like systems:  
    1.Download the model GraphDef file from:  
      http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt  
    2.Download 12 checkpoint shared files from:  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7  
      http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8  

   
To generate the Language Model Intermediate Representation (IR), provide TensorFlow Language Model to the Model Optimizer with parameters:  

```sh
python3 ./mo_tf.py
--input_model lm_1b/graph-2016-09-10.pbtxt                        \
--input_checkpoint lm_1b/ckpt                                     \
--input_model_is_text                                             \
--output softmax_out,,lstm/lstm_0/concat_2,lstm/lstm_1/concat_2   \
--input_shape [50],[1,9216],[1,9216]                              \
--input 0:char_embedding/Reshape,Variable/read,Variable_1/read    \
```

Running  
-------
Running the application with the `-h` option yields the following usage message:

```
usage: lm_1b_sample.py [-h] -m MODEL -i INPUT -v VOCAB [-l CPU_EXTENSION]
                    [-d DEVICE] [-n NUMBER_SAMPLES]

Options:
  -h, --help            Show this help message and exit.
  -m MODEL, --model MODEL
                        Required. Path to an .xml file with a trained model.
  -i INPUT, --input INPUT
                        Required. The prefix used for word predictions
  -v VOCAB, --vocab VOCAB
                        Required. The vocab file
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        Optional. Required for CPU custom layers. MKLDNN
                        (CPU)-targeted custom layers. Absolute path to a
                        shared library with the kernels implementations.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, FPGA, HDDL, MYRIAD or HETERO: is acceptable. The
                        sample will look for a suitable plugin for device
                        specified. Default value is CPU
  -n NUMBER_SAMPLES, --number_samples NUMBER_SAMPLES
                        Optional. Set number of samples. number of samples
```

Running Demo

```sh
python lm_1b_sample.py 
       -m path_to_IR_model/graph-2016-09-10.xml 
       -i 'What is' 
       -v path_to_Vocabulary_file/vocab-2016-09-10.txt 
```
The vocabulary file can be downloaded from http://download.tensorflow.org/models/LM_LSTM_CNN/vocab-2016-09-10.txt

Demo Output
------------
The application shows the predicted sentence:
```
What
What is
What is your
What is your relationship
What is your relationship with the 
...(omitted)
```
