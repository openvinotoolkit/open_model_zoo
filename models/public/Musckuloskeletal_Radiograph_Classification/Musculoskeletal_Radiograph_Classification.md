
# **Abnormality and Joint Type Detection in Musculoskeletal Radiographs**

## **Use Case and High-Level Description**

We are releasing the CNN models which were trained to perform a systematic search over different CNN architectures for screening Musculoskeletal Radiographs. 6 CNN architectures have been trained under different training protocols for the joint tasks of identifying the joint/bone type in the X-ray image as well as detecting the presence of any abnormalities (such as fractures ) in it.

The CNN architectures have been modified by adding a convolutional layer in the beginning to convert the grayscale input to three channels. At the end of the CNN, the Fully Connected layers have also been modified to obtain two predictions (one for the type of joint/bone present in the image and the other for abnormality prediction). _In this repository, we provide the ONNX models (obtained from our Pytorch implementation) for the Inception v3, ResNet-18, Xception, DenseNet-121, VGG-16 and VGG-19 architectures which were initialized with the imagenet pre-trained weights and fine-tuned using an unweighted categorical cross-entropy loss._

The details of the CNN model and our experiments are reported in [1].

[1] A. Chakravarty, N.Ghosh, D.Sheet, T.Sarkar, R. Sethuraman, &quot;Radiologist Validated Systematic Search over DeepNeural Networks for Screening MusculoskeletalRadiographs&quot;, MedNeuRIPS Workshop, 2019 (url:[https://profs.etsmtl.ca/hlombaert/public/medneurips2019/66\_CameraReadySubmission\_Musculoskeletal\_cameraready\_MedNeurIPS.pdf](https://profs.etsmtl.ca/hlombaert/public/medneurips2019/66_CameraReadySubmission_Musculoskeletal_cameraready_MedNeurIPS.pdf) )

## **Dataset**

The experiments have been performed using the publicly available MURA dataset [2] which can be downloaded from :[https://stanfordmlgroup.github.io/competitions/mura/](https://stanfordmlgroup.github.io/competitions/mura/)

[2] Rajpurkar, Pranav, Jeremy Irvin, Aarti Bagul, Daisy Ding, Tony Duan, Hershel Mehta, Brandon Yang et al. &quot;Mura: Large dataset for abnormality detection in musculoskeletal radiographs.&quot; _arXiv preprint arXiv:1712.06957_ (2017).

## **Example**

An example for using the onnx models shared in this repository is available here:

[https://drive.google.com/file/d/1lr6iEe7Wc3i2g-Cf6UPU8AaQRhUs7MrA/view?usp=sharing](https://drive.google.com/file/d/1lr6iEe7Wc3i2g-Cf6UPU8AaQRhUs7MrA/view?usp=sharing)

## **Specification**

| Type | Classification (Mult-task) |
| --- | --- |
| Source Framework | Pytorch (converted to ONNX) |

In this work, we compared the performance of different CNN architectures. The ONNX model for each of them is provided inside the corresponding sub-directories.

|
 | Inception v3 | Xception | DenseNet-121 | ResNet-18 | VGG16 | VGG19 |
| --- | --- | --- | --- | --- | --- | --- |
| GFLOPS\* | 5.73 | 8.44 | 2.90 | 1.83 | 15.53 | 19.69 |
| Mparams | 21.991 | 21.01 | 7.057 | 11.229 | 134.680 | 139.992 |

\* The GFLOPs have been computed using the Library : [https://pypi.org/project/thop/](https://pypi.org/project/thop/) on our original implementation in Pytorch and may vary from the actual FLOPs in the released ONNX models provided.

## **Performance**

## A systematic evaluation of 6 CNN architectures indicated that:

- All 6 CNN architectures performed very well on the task of the Joint/Bone type detection with an Average Area under the ROC curve (AUROC) in the range of [0.969, 0.992] and an average accuracy in the range of [0.955, 0.995] across the different architectures.
- The abnormality detection task is relatively more challenging with the AUROC varying in the range of [0.752, 0.880] and the accuracy in the range of [0.737, 0.835] across the 6 CNN architectures.
- The Xception architecture was found to perform the best (Avg. AUROC of 0.992 for joint type and Avg AUROC of 0.877 for abnormality detection ) closely followed by ResNet-18 (Avg. AUROC of 0.994 for joint type and Avg AUROC of 0.868 for abnormality detection )

Further details of the performance are available in our workshop paper ( [https://profs.etsmtl.ca/hlombaert/public/medneurips2019/66\_CameraReadySubmission\_Musculoskeletal\_cameraready\_MedNeurIPS.pdf](https://profs.etsmtl.ca/hlombaert/public/medneurips2019/66_CameraReadySubmission_Musculoskeletal_cameraready_MedNeurIPS.pdf) ) and it&#39;s Supplementary Material ( [http://bit.do/Supplementary-MedNeurIPS2019](http://bit.do/Supplementary-MedNeurIPS2019))

## **Input**

For both the original model in Pytorch as well as the ONNX model provided in this repository, the input is a grayscale chest radiograph. The input image is resized to (1,1,299,299) for the Inception and the Xception Models. For all other models, the input is resized to (1,1,224,224). The order of the dimensions for the input to the ONNX model is (B,C,H,W) where B is the batchsize, C is the number of channels and H,W are the spatial dimensions of the image.

We refer to our example code for further details on preprocessing [https://drive.google.com/file/d/1lr6iEe7Wc3i2g-Cf6UPU8AaQRhUs7MrA/view?usp=sharing](https://drive.google.com/file/d/1lr6iEe7Wc3i2g-Cf6UPU8AaQRhUs7MrA/view?usp=sharing) .

## **Output**

For both the original model in Pytorch as well as the ONNX model provided in this repository, the CNN has two outputs: pred1, pred2=model(image)

Here pred1 is of size (B,2) and provides the prediction score for the presence of an abnormality in the input image. In our case, the batch size B=1.

Similarly, pred2 is of size (B, 7) which consists of the prediction scores for the seven joint/bone type present in the given image. The class labels corresponding to the 7 joint/bone types are as follows:

0: Humerus

1: Wrist

2: Forearm

3:Finger

4: Hand

5: Elbow

6: Shoulder

The prediction labels for the two tasks can be obtained as argmax(pred1, axis=1) and argmax(pred2, axis=1) respectively.

# Legal Information

The model is distributed under theApache License, Version 2.0

Copyright (c) 2020 Debdoot Sheet

Licensed under the Apache License, Version 2.0 (the &quot;License&quot;);

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an &quot;AS IS&quot; BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.
