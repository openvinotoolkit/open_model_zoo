# BERT model Python* Demo

## BERT Introduction
BERT model is developed by Google, it's a new method of pre-praining language represenatations which performs state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks, for details please find it on [github](https://github.com/google-research/bert).

## Download model and Convert to IR
Download pre-trained BERT model from this [link](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip).

For how to convert TensorFlow BERT model to IR, please follow the user guide in [OpenVINO](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_BERT_From_Tensorflow.html).

## How It Works
BERT model is some kinds of word2vec, its output is a vector. For specific application, general principle is to cascade a network with BERT model, and fine-tune whole model to get better performance. Here we just use the pre-trained BERT model and demonstrate the usage and function of BERT,for using fine-tuned BERT model to do infernce, please go to directory "fine-tune_bert_demo".

Here are two sample codes for BERT model, one is encoding, it encodes a sentence to a vector, the sentence will go through a series of pre-process and get 3 feature vectors which dimension are all 1\*128, then the feature vectors are fed into BERT model, and BERT model output a vector, which dimension is 1\*768.

The second sample code is similar with the [repo](https://github.com/hanxiao/bert-as-service), use BERT to build a QA semantic search engine. The sample receives a sentence as input, and it outputs 5 most similar sentences in the registered questions.txt file.
Since the number of registered sentences is small and the BERT is not fine-tuned, it'll not always work well on any input sentence.


>**Note**:some of the codes are borrowed from Google, for license please follow Google's policy.

## Running
Please install OpenVINO and source the OPENVINO_INSTALLDIR/openvino/bin/setupvars.sh. We tested these sample code on OpenVINO 2019R3.
* BERT encoding sample

  The command is:

  `python3 bert_encoding_demo.py -m path_to_model/bert_model.ckpt.xml -l path_to_cpu_extension/libcpu_extension.so -d CPU`

  For example:

  `python3 bert_encoding_demo.py -m bert_model.ckpt.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx512.so -d CPU`


* BERT semantic search sample

  The command is same as above:

  `python3 bert_semantic_search_demo.py -m path_to_model/bert_model.ckpt.xml -l path_to_cpu_extension/libcpu_extension.so -d CPU`
  
## Demo Output
* For bert_encoding_demo, it will output values of an array in 1*768 dimension, which is original BERT output, kind of word embedding, feature encoding/vector.
* For bert_semantic_search_demo, it will get a sentences as input and output most 5 matched sentences.
