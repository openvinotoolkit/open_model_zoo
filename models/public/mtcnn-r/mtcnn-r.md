# mtcnn-r

## Use Case and High-Level Description

The `mtcnn-r` model is one of the [mtcnn](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) group of models designed to perform face detection. Short for "Multi-task Cascaded Convolutional Neural Network", it is implemented using the Caffe framework. The "r" designation indicates that this model is the "refine" network intended to refine the data returned as output from the "proposal" `mtcnn-p` network. For details about this family of models, check out the [repository](https://github.com/DuinoDu/mtcnn).

The model input is a blob with a vector containing the first pass of face data, as returned by the `mtcnn-p` model.

The model output is a blob with a vector containing the refined face data. If there are no faces detected by the refine pass, no further processing is needed. Otherwise, you will typically use this output as input to the `mtcnn-o` model.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 0.003         |
| MParams           | 0.1           |
| Source framework  | Caffe         |

## Accuracy

## Performance

## Input

Name - `data`, shape - `1,3,24,24`

## Output

Name: `prob1`

## Legal Information

[https://raw.githubusercontent.com/DuinoDu/mtcnn/master/LICENSE]()
