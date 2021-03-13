# stardist_dsb2018

## Use Case and High-Level Description

[StarDist](https://github.com/stardist/stardist) model trained on fluorescence (single channel) images to detect nuclei.

## Input

### Original Model

One tile of the shape [1x1024x1024x1] in the [BxHxWxC] format, where:

- `B` - batch size
- `H` - height of tile
- `W` - width of tile
- `C` - channel

### Converted Model

One image of the shape [1x1x1024x1024] in the [BxCxHxW] format, where:

- `B` - batch size
- `C` - channel
- `H` - height of tile
- `W` - width of tile

## Output

Network produces a concatenated tensor of shape [1x33xHxW] where first channel is probabilities
of detected objects with a center (`x`, `y`) for an every `x in [0, W)` and `y in [0, H)`. Values
are in range `[0, 1]`. And the rest `32` channels are distances among base angles for every detected
object with a center in that position. Spatial dimensions of output tensor match input sizes.

### Original Model

Blob of the shape [1, 1024, 1024, 33] in the [BxHxWxC] format, where:

- `B` - batch size
- `H` - height of tile
- `W` - width of tile
- `C` - probabilities (1 channel) and distances (32 channels)

### Converted Model

Blob of the shape [1, 33, 1024, 1024] in the [BxCxHxW] format, where:

- `B` - batch size
- `C` - probabilities (1 channel) and distances (32 channels)
- `H` - height of tile
- `W` - width of tile


## Legal Information
[\*] Other names and brands may be claimed as the property of others.

The original model is distributed under the
[MIT License](https://raw.githubusercontent.com/stardist/stardist-imagej/master/LICENSE.txt).

```
BSD 3-Clause License

Copyright (c) 2019, Uwe Schmidt, Martin Weigert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
