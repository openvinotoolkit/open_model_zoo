# landmarks-regression-retail-0001

## Use Case and High-Level Description

This is a lightweight landmarks regressor for a Smart Classroom scenario. It has a classic convolutional design: stacked 3x3 convolutions, batch normalizations, ELU activations, and poolings. Final regression is done by the global depthwise pooling head and FullyConnected layers. The model predicts five facial landmarks: two eyes, nose, and two lip corners.

## Example

![](./landmarks-regression-retail-0001.png)

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| RMSD (on VGGFace2)              | 0.0897                                    |
| Face location requirements      | Tight crop                                |
| GFlops                          | 0.02                                      |
| MParams                         | 0.2                                       |
| Source framework                | Caffe*                                    |

[RMSD](https://en.wikipedia.org/wiki/Root-mean-square_deviation) is a root-mean-square deviation.

## Inputs

1. Name: "data" , shape: [1x3x48x48] - An input image in the format [BxCxHxW],
   where:
    - B - batch size
    - C - number of channels
    - H - image height
    - W - image width

   The expected color order is RGB.

## Outputs


1.	The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values 
	for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5). 
	All the coordinates are normalized to be in range [0,1].

	-	Output layer name in Inference Engine format: 
		`landmarks_xy`

	-	Output layer name in Inference Caffe* format: 
		`landmarks_xy`


## Legal Information
[*] Other names and brands may be claimed as the property of others.
