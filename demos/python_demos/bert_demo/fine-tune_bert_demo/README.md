# Fine-tuned BERT model Python* Demo

## How It Works
This sample code show how to run fine-tuned BERT model using OpenVINO toolkit on a simple specific task to predict whether an IMDB movie review is positive or negative.

For how to fine-tune BERT model and use OpenVINO to convert BERT model, please go to directory "bert_training_fine-tune" and follow the guide.

## Running
After get converted fine-tuned BERT model, run below command to infer, please change the libcpu_extension.so path if needed on your machine.

  `python3 bert_fine-tune_for_movie-review_demo.py -m path_to_bert-model/bert-finetune.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_avx512.so -d CPU`
