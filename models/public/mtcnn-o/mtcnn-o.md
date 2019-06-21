# mtcnn-o

## Use Case and High-Level Description

The `mtcnn-o` model is the third of the [mtcnn](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) group of models designed to perform face detection. Short for "Multi-task Cascaded Convolutional Neural Network", it is implemented using the Caffe framework. The "o" designation indicates that this model is the "output" network intended to take the data returned from the "refine" `mtcnn-r` network, and transform it into the final output data.  For details about this family of models, check out the [repository](https://github.com/DuinoDu/mtcnn).

The model input is a blob with a vector containing the refined face data, as returned by the `mtcnn-r` model.

The model output is a blob with a vector containing the output face data.

## Example

## Specification

| Metric            | Value         |
|-------------------|---------------|
| Type              | Detection     |
| GFLOPs            | 0.026         |
| MParams           | 0.389         |
| Source framework  | Caffe         |

## Accuracy

## Performance

## Input

Name - `data`, shape - `1,3,48,48`

## Output

Name: `prob1`

## Legal Information

[https://raw.githubusercontent.com/DuinoDu/mtcnn/master/LICENSE]()
