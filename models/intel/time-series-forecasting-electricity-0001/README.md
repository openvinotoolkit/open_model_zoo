# time-series-forecasting-electricity-0001

## Use Case and High-Level Description

This is a Time Series Forecasting model based on the Temporal Fusion Transformer and model trained on the Electricity dataset.

## Specification

| Metric            | Value                 |
|-------------------|-----------------------|
| GOps              | 0.40                  |
| MParams           | 2.26                  |
| Source framework  | PyTorch\*             |

## Accuracy

| Metric                          | Value         |
|---------------------------------|---------------|
| Normalized Quantile Loss (P50)  |        0.056  |
| Normalized Quantile Loss (P90)  |        0.028  |

Normalized Quantile Loss described in [Bryan Lim et al.](https://arxiv.org/abs/1912.09363).

The quality metrics were calculated on the Electricity dataset (test split).

## Input

name: `timestamps`
shape: `1, 192, 5`
format: `B, T, N`
`B` - batch size.
`T` - number of input timestamps.
`N` - number of input features.

## Output

name: `quantiles`
shape: `1, 24, 3`
format: `B, T, Q`
`B` - batch size.
`T` - number of output timestamps.
`Q` - number of output quantiles (0.1, 0.5, 0.9).

## Legal Information
[*] Other names and brands may be claimed as the property of others.
