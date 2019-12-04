# Fine-tune BERT model

## Introduction
Here we will show how to fine-tune BERT model on a simple specific task to predict whether an IMDB movie review is positive or negative, and then run the inference using Intel OpenVINO toolkit. The original code comes from this [link](https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb), you can also fine-tune BERT model on Google [colab](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb). Here We fine-tine BERT on local machine with TITAN GPU, you can also fine-tune using CPU within a few epochs, but it may take longer time.

## Preparation
The fine-tune process was done on TensorFlow with version 1.12.0, higher version is ok but maybe lead to incompatibility for some TensorFlow API.


## Fine-tune BERT model
1. Download pre-trained [BERT model](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip), put it in directory "pre-train-model" and unzip it.

2. Export pre-trained model path

     `export BERT_BASE_DIR = path_to_model/pre-train-model/multilingual_L-12_H-768_A-12`

3. Train BERT model with few epochs

   ```
   python3 run_classifier.py  --do_train=true   \
   --do_eval=true   \
   --vocab_file=$BERT_BASE_DIR/vocab.txt   \
   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   \
   --max_seq_length=128   \
   --train_batch_size=32   \
   --learning_rate=2e-5   \
   --num_train_epochs=3.0 \
   --output_dir=./result1/
   ```

   The fine-tuned model will be saved in directory "result1", the accuracy is about 0.8436. Since by default the fine-tuned model contains dropout layer which can't be converted by OpenVINO toolkit, we need to remove the dropout layer, so we fine-tune again BERT model without dropout.

4. Fine-tune previous BERT model from directory "result1" without dropout layer

   (1). Modify code to disable dropout layer

        in run_classifier.py, modify code as blow to make flag "is_training" is False and save file.
        
        #is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        is_training = False
        

   (2). Fine-tune model, init_checkpoint comes from directory "result1", and save fine-tuned mode in directory "result2".

         
         python3 run_classifier.py  --do_train=true   --do_eval=true   \
         --vocab_file=$BERT_BASE_DIR/vocab.txt   \
         --bert_config_file=$BERT_BASE_DIR/bert_config.json   \
         --init_checkpoint=./result1/model.ckpt-468   \
         --max_seq_length=128   \
         --train_batch_size=32   \
         --learning_rate=2e-5   \
         --num_train_epochs=3.0   \
         --output_dir=./result2/
         

5. Run prediction with fine-tune BERT model.

    ```
    python3 run_classifier_predcit.py  \
    --do_train=false   \
    --do_eval=false \
    --do_predict=true \
    --vocab_file=$BERT_BASE_DIR/vocab.txt   \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=result2/model.ckpt-468 \
    --output_dir=result2/
    ```

    You'll find predictions for 4 movie reviews like this:
     ```
     ('That movie was absolutely awful', array([0.9951434 , 0.00485664], dtype=float32), 'Negative')
     ('The acting was a bit lacking', array([0.994231  , 0.00576904], dtype=float32), 'Negative')
     ('The film was creative and surprising', array([0.00695132, 0.99304867], dtype=float32), 'Positive')
     ('Absolutely fantastic!', array([0.00698267, 0.9930173 ], dtype=float32), 'Positive')
     ```

## Convert Fine-tuned BERT model using OpenVINO
The fine-tuned model doesn't save topology for the fully connected layer in checkpoint file, so we export BERT model and cascaded fully connected layer separately.

1. Convert checkpoint to .pb foramt, removing many unneeded nodes, total 1636 ops are saved, after conversion, we will get bert-finetune.pb

   `python3 ckt_to_pb.py`

2. Export weight and bias for fully connected layer, the weight and bias will be saved in file "weight.npy" and "bias.npy" respectively.

    `python3 export_fc.py`

3. Convert BERT model using OpenVINO

   In OpenVINO MO path, such as "/opt/intel/openvino/deployment_tools/model_optimizer",
   execute below command to convert .pb format BERT model:

   `python3 ./mo_tf.py --input_model path_to_bert_model/bert-finetune.pb --input IteratorGetNext:0,IteratorGetNext:1,IteratorGetNext:3 --input_shape [1,128],[1,128],[1,128] --output bert/pooler/dense/Tanh  --disable_nhwc_to_nchw`

## Run inference using OpenVINO
Copy converted BERT model, weight.npy and bias,npy to directory "python_sample_for_fine-tune_bert", and run below command to infer, please change the libcpu_extension.so path if needed on your machine.

  `python3 bert_fine-tune_for_movie-review.py -m path_to_bert-model/bert-finetune.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx512.so -d CPU`
