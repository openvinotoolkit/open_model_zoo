## **Abnormality Detection in Chest Radiographs**

## **Use Case and High-Level Description**

We are releasing two trained CNN models to detect the presence of fourteen disease classes in 2D Chest X-ray radiograph images. The first model employs the DenseNet-121 [1] CNN architecture which has been shown to have a good performance on this task in existing literature [3], [4]. The second CNN model was obtained by performing a systematic search over a set of CNN architectures obtained by scaling the width and depth of DenseNet using the EfficientNet method [2]. The optimal architecture was achieved with the scaling parameters of. The details of our experiments based on EfficientNet is summarized in the slides available at: [https://drive.google.com/file/d/1BQsCIeYFHZlMGmUS5GYIYQmgiMqPwPc4/view?usp=sharing](https://drive.google.com/file/d/1BQsCIeYFHZlMGmUS5GYIYQmgiMqPwPc4/view?usp=sharing)

[1] Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. &quot;Densely connected convolutional networks.&quot; In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp. 4700-4708. 2017.

[2] Tan, Mingxing, and Quoc V. Le. &quot;EfficientNet: Rethinking model scaling for convolutional neural networks.&quot; , ICML, pp. 6105-6114. 2019.

[3] Mitra, Arka, Arunava Chakravarty, Nirmalya Ghosh, Tandra Sarkar, Ramanathan Sethuraman, and Debdoot Sheet. &quot;A Systematic Search over Deep Convolutional Neural Network Architectures for Screening Chest Radiographs.&quot; _arXiv preprint arXiv:2004.11693_ (2020).

## **Dataset**

The experiments have been performed using the publicly available CheXpert dataset [4]. The **download link and the License of the public dataset**  can be found at: [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/).

Alternative chest X-ray dataset for testing and evaluating can be found at [https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) and [https://data.mendeley.com/datasets/rscbjbr9sj/2](https://data.mendeley.com/datasets/rscbjbr9sj/2).

[4] Irvin, Jeremy, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund et al. &quot;Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison.&quot; In _Proceedings of the AAAI Conference on Artificial Intelligence_, vol. 33, pp. 590-597. 2019.

##


## **Example**

An example for using the onnx models shared in this repository is available here:

[https://drive.google.com/file/d/1FCG9EaslKa\_n\_OF6FpcPlNy9Q3NNY4Rn/view?usp=sharing](https://drive.google.com/file/d/1FCG9EaslKa_n_OF6FpcPlNy9Q3NNY4Rn/view?usp=sharing)

## **Specification and Performance**

The DenseNet-121 architecture and the optimized DenseNet CNN model obtained through EfficientNet based architecture search is provided in the

corresponding sub-directories named &quot;baseline\_DenseNet121&quot; and &quot;EfficientNet\_DenseNet&quot; respectively. The performance is measured using the average Area under the ROC curves (AUROC) across fourteen disease classes.

| Type | Classification (Mult-task) |
| --- | --- |
| Source Framework | Pytorch (converted to ONNX) |


|     | DenseNet-121 | DenseNet-EfficientNet |
| --- | --- | --- |
| GFLOPs | 5.88 | 6.85 |
| Mparams | 6.97 | 7.13 |
| Avg. AUROC | 0.7477 | 0.8010 |

**Input**

For both the DenseNet-121 and the optimal DenseNet model in the ONNX model provided in this repository, the input grayscale image is resized to 320 X 320 and replicated to three channels. Each of the 3 image channels are normalized to match the distribution of the Image-Net dataset. The input is resized to (1,1,320,320). The order of the dimensions for the input to the ONNX model is (B,C,H,W) where B is the batchsize, C is the number of channels and H,W are the spatial dimensions of the image.

The details of the image preprocessing steps is detailed in our demo code available at: [https://drive.google.com/file/d/1FCG9EaslKa\_n\_OF6FpcPlNy9Q3NNY4Rn/view?usp=sharing](https://drive.google.com/file/d/1FCG9EaslKa_n_OF6FpcPlNy9Q3NNY4Rn/view?usp=sharing)

## **Output**

The ONNX models provided in this repository, outputs the prediction scores in the range of [0,1] for the fourteen disease classes. We pose the Chest radiograph screening as a multi-label classification task where multiple disease classes can co-occur in the same image. The ordering of the disease classes is listed below:

| Class Label dimension | Disease Class |
| --- | --- |
| 0 | _&#39;No Finding&#39;_ |
| 1 | _&#39;Enlarged Cardiomediastinum&#39;_ |
| 2 | _&#39;Cardiomegaly&#39;_ |
| 3 | _&#39;Lung Opacity&#39;_ |
| 4 | _&#39;Lung Lesion&#39;_ |
| 5 | _&#39;Edema&#39;_ |
| 6 | _&#39;Consolidation&#39;_ |
| 7 | _&#39;Pneumonia&#39;_ |
| 8 | _&#39;Atelectasis&#39;_ |
| 9 | _&#39;Pneumothorax&#39;_ |
| 10 | _&#39;Pleural Effusion&#39;_ |
| 11 | _&#39;Pleural Other&#39;_ |
| 12 | _&#39;Fracture&#39;_ |
| 13 | &#39;_Support Devices&#39;_ |

**Legal Information**

The model is distributed under theApache License, Version 2.0

Copyright (c) 2020 Debdoot Sheet, Ramanathan Sethuraman

Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an &quot;AS IS&quot; BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.
