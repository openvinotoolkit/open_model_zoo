"""
 Copyright (C) 2023 KNS Group LLC (YADRO)
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

HDR_Y_MAX = 65504.  # maximum HDR value


def luminance(r, g, b):
    return 0.212671 * r + 0.715160 * g + 0.072169 * b


def autoexposure(image):
    key = 0.18
    eps = 1e-8
    k = 16  # downsampling amount

    # Compute the luminance of each pixel
    r = image[..., 0]
    g = image[..., 1]
    b = image[..., 2]
    lum = luminance(r, g, b)

    # Down sample the image to minimize sensitivity to noise
    h = lum.shape[0]  # original height
    w = lum.shape[1]  # original width
    hk = (h + k // 2) // k  # down sampled height
    wk = (w + k // 2) // k  # down sampled width

    lk = np.zeros((hk, wk), dtype=lum.dtype)
    for i in range(hk):
        for j in range(wk):
            begin_h = i * h // hk
            begin_w = j * w // wk
            end_h = (i + 1) * h // hk
            end_w = (j + 1) * w // wk

            lk[i, j] = lum[begin_h:end_h, begin_w:end_w].mean()

    lum = lk

    # Keep only values greater than epsilon
    lum = lum[lum > eps]
    if lum.size == 0:
        return 1.

    # Compute the exposure value
    return float(key / np.exp2(np.log2(lum).mean()))


def round_up(a, b):
    return (a + b - 1) // b * b


def get_transfer_function():
    return PUTransferFunction()


# Transfer function: sRGB

SRGB_A = 12.92
SRGB_B = 1.055
SRGB_C = 1. / 2.4
SRGB_D = -0.055
SRGB_Y0 = 0.0031308
SRGB_X0 = 0.04045


def srgb_forward(y):
    return np.where(y <= SRGB_Y0,
                    SRGB_A * y,
                    SRGB_B * np.power(y, SRGB_C) + SRGB_D)


def srgb_inverse(x):
    return np.where(x <= SRGB_X0,
                    x / SRGB_A,
                    np.power((x - SRGB_D) / SRGB_B, 1. / SRGB_C))


class SRGBTransferFunction:
    def forward(self, y):
        return srgb_forward(y)

    def inverse(self, x):
        return srgb_inverse(x)


# Transfer function: PU

# Fit of PU2 curve normalized at 100 cd/m^2
# [Aydin et al., 2008, "Extending Quality Metrics to Full Luminance Range Images"]
PU_A = 1.41283765e+03
PU_B = 1.64593172e+00
PU_C = 4.31384981e-01
PU_D = -2.94139609e-03
PU_E = 1.92653254e-01
PU_F = 6.26026094e-03
PU_G = 9.98620152e-01
PU_Y0 = 1.57945760e-06
PU_Y1 = 3.22087631e-02
PU_X0 = 2.23151711e-03
PU_X1 = 3.70974749e-01


def pu_forward(y):
    return np.where(y <= PU_Y0,
                    PU_A * y,
                    np.where(y <= PU_Y1,
                             PU_B * np.power(y, PU_C) + PU_D,
                             PU_E * np.log(y + PU_F) + PU_G))


def pu_inverse(x):
    return np.where(x <= PU_X0,
                    x / PU_A,
                    np.where(x <= PU_X1,
                             np.power((x - PU_D) / PU_B, 1. / PU_C),
                             np.exp((x - PU_G) / PU_E) - PU_F))


PU_NORM_SCALE = 1. / pu_forward(HDR_Y_MAX)


class PUTransferFunction:
    def forward(self, y):
        return pu_forward(y) * PU_NORM_SCALE

    def inverse(self, x):
        return pu_inverse(x / PU_NORM_SCALE)
