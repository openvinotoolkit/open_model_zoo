"""
Copyright (c) 2018-2021 Intel Corporation
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
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..config import NumberField, StringField, BoolField
from .regression import BaseRegressionMetric
from ..representation import (
    ImageInpaintingAnnotation, ImageProcessingAnnotation,
    SuperResolutionAnnotation, StyleTransferAnnotation, ImageInpaintingPrediction, ImageProcessingPrediction,
    SuperResolutionPrediction, StyleTransferPrediction
)
from ..utils import UnsupportedPackage

try:
    from scipy.signal import convolve2d
except ImportError as import_err:
    convolve2d = UnsupportedPackage('scipy', import_err)

try:
    import lpips
except ImportError as import_err:
    lpips = UnsupportedPackage('lpips', import_err)


def _ssim(annotation_image, prediction_image):
    prediction = np.asarray(prediction_image)
    ground_truth = np.asarray(annotation_image)
    if len(ground_truth.shape) < len(prediction.shape) and prediction.shape[-1] == 1:
        prediction = np.squeeze(prediction)
    mu_x = np.mean(prediction)
    mu_y = np.mean(ground_truth)
    var_x = np.var(prediction)
    var_y = np.var(ground_truth)
    sig_xy = np.mean((prediction - mu_x)*(ground_truth - mu_y))
    c1 = (0.01 * 2**8-1)**2
    c2 = (0.03 * 2**8-1)**2
    mssim = (2*mu_x*mu_y + c1)*(2*sig_xy + c2)/((mu_x**2 + mu_y**2 + c1)*(var_x + var_y + c2))
    return mssim


class StructuralSimilarity(BaseRegressionMetric):
    __provider__ = 'ssim'
    annotation_types = (ImageInpaintingAnnotation, ImageProcessingAnnotation, SuperResolutionAnnotation,
                        StyleTransferAnnotation)
    prediction_types = (ImageInpaintingPrediction, ImageProcessingPrediction, SuperResolutionPrediction,
                        StyleTransferPrediction)

    def __init__(self, *args, **kwargs):
        super().__init__(_ssim, *args, **kwargs)
        self.meta['target'] = 'higher-better'
        self.meta['target_per_value'] = {'mean': 'higher-better', 'std': 'higher-worse', 'max_error': 'higher-worse'}


class PeakSignalToNoiseRatio(BaseRegressionMetric):
    __provider__ = 'psnr'

    annotation_types = (SuperResolutionAnnotation, ImageInpaintingAnnotation, ImageProcessingAnnotation,
                        StyleTransferAnnotation)
    prediction_types = (SuperResolutionPrediction, ImageInpaintingPrediction, ImageProcessingPrediction,
                        StyleTransferPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'scale_border': NumberField(
                optional=True, min_value=0, default=4, description="Scale border.", value_type=int
            ),
            'color_order': StringField(
                optional=True, choices=['BGR', 'RGB'], default='RGB',
                description="The field specified which color order BGR or RGB will be used during metric calculation."
            ),
            'normalized_images': BoolField(optional=True, default=False, description='images in [0, 1] range or not')
        })

        return parameters

    def __init__(self, *args, **kwargs):
        super().__init__(self._psnr_differ, *args, **kwargs)
        self.meta['target'] = 'higher-better'
        self.meta['target_per_value'] = {'mean': 'higher-better', 'std': 'higher-worse', 'max_error': 'higher-worse'}

    def configure(self):
        super().configure()
        self.scale_border = self.get_value_from_config('scale_border')
        self.color_order = self.get_value_from_config('color_order')
        channel_order = {
            'BGR': [2, 1, 0],
            'RGB': [0, 1, 2],
        }
        self.meta['postfix'] = 'Db'
        self.channel_order = channel_order[self.color_order]
        self.normalized_images = self.get_value_from_config('normalized_images')
        self.color_scale = 255 if not self.normalized_images else 1

    def _psnr_differ(self, annotation_image, prediction_image):
        prediction = np.squeeze(np.asarray(prediction_image)).astype(np.float)
        ground_truth = np.squeeze(np.asarray(annotation_image)).astype(np.float)

        height, width = prediction.shape[:2]
        prediction = prediction[
            self.scale_border:height - self.scale_border,
            self.scale_border:width - self.scale_border
        ]
        ground_truth = ground_truth[
            self.scale_border:height - self.scale_border,
            self.scale_border:width - self.scale_border
        ]
        image_difference = (prediction - ground_truth) / self.color_scale
        if len(ground_truth.shape) == 3 and ground_truth.shape[2] == 3:
            r_channel_diff = image_difference[:, :, self.channel_order[0]]
            g_channel_diff = image_difference[:, :, self.channel_order[1]]
            b_channel_diff = image_difference[:, :, self.channel_order[2]]

            channels_diff = (r_channel_diff * 65.738 + g_channel_diff * 129.057 + b_channel_diff * 25.064) / 256

            mse = np.mean(channels_diff ** 2)
            if mse == 0:
                return np.Infinity
        else:
            mse = np.mean(image_difference ** 2)

        return -10 * math.log10(mse)


class PeakSignalToNoiseRatioWithBlockingEffectFactor(PeakSignalToNoiseRatio):
    __provider__ = 'psnr-b'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'block_size': NumberField(value_type=int, min_value=1, default=8, description='block size for BEF')
        })
        return params

    def configure(self):
        super().configure()
        self.block_size = self.get_value_from_config('block_size')

    def _psnr_differ(self, annotation_image, prediction_image):
        prediction = np.asarray(prediction_image).astype(np.float)
        ground_truth = np.asarray(annotation_image).astype(np.float)

        height, width = prediction.shape[:2]
        prediction = prediction[
            self.scale_border:height - self.scale_border,
            self.scale_border:width - self.scale_border
        ]
        ground_truth = ground_truth[
            self.scale_border:height - self.scale_border,
            self.scale_border:width - self.scale_border
        ]

        if prediction.ndim > 2 and prediction.shape[-1] == 3:
            if self.color_order == 'BGR':
                ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2YCR_CB)
                prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2YCR_CB)

        return self._psnrb(ground_truth, prediction)

    def _psnrb(self, gt, pred):
        if gt.ndim == 3:
            gt = gt[:, :, 0]

        if pred.ndim == 3:
            pred = pred[:, :, 0]

        imdff = gt.astype(np.double) - pred.astype(np.double)

        mse = np.mean(np.square(imdff.flatten()))
        bef = self._compute_bef(pred, block_size=self.block_size)
        mse_b = mse + bef

        if np.amax(pred) > 2:
            psnr_b = 10 * math.log10(255 ** 2 / mse_b)
        else:
            psnr_b = 10 * math.log10(1 / mse_b)

        return psnr_b

    @staticmethod
    def _compute_bef(im, block_size=8):
        if len(im.shape) == 3:
            height, width, channels = im.shape
        elif len(im.shape) == 2:
            height, width = im.shape
            channels = 1
        else:
            raise ValueError("Not a 1-channel/3-channel grayscale image")

        if channels > 1:
            raise ValueError("Not for color images")

        h = np.array(range(0, width - 1))
        h_b = np.array(range(block_size - 1, width - 1, block_size))
        h_bc = np.array(list(set(h).symmetric_difference(h_b)))

        v = np.array(range(0, height - 1))
        v_b = np.array(range(block_size - 1, height - 1, block_size))
        v_bc = np.array(list(set(v).symmetric_difference(v_b)))

        d_b = 0
        d_bc = 0

        # h_b for loop
        for i in list(h_b):
            diff = im[:, i] - im[:, i + 1]
            d_b += np.sum(np.square(diff))

        # h_bc for loop
        for i in list(h_bc):
            diff = im[:, i] - im[:, i + 1]
            d_bc += np.sum(np.square(diff))

        # v_b for loop
        for j in list(v_b):
            diff = im[j, :] - im[j + 1, :]
            d_b += np.sum(np.square(diff))

        # V_bc for loop
        for j in list(v_bc):
            diff = im[j, :] - im[j + 1, :]
            d_bc += np.sum(np.square(diff))

        # N code
        n_hb = height * (width / block_size) - 1
        n_hbc = (height * (width - 1)) - n_hb
        n_vb = width * (height / block_size) - 1
        n_vbc = (width * (height - 1)) - n_vb

        # D code
        d_b /= (n_hb + n_vb)
        d_bc /= (n_hbc + n_vbc)

        # Log
        if d_b > d_bc:
            t = np.log2(block_size) / np.log2(min(height, width))
        else:
            t = 0

        # BEF
        bef = t * (d_b - d_bc)

        return bef


class VisionInformationFidelity(BaseRegressionMetric):
    __provider__ = 'vif'

    annotation_types = (SuperResolutionAnnotation, ImageInpaintingAnnotation, ImageProcessingAnnotation,
                        StyleTransferAnnotation)
    prediction_types = (SuperResolutionPrediction, ImageInpaintingPrediction, ImageProcessingPrediction,
                        StyleTransferPrediction)

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'sigma_nsq': NumberField(
                value_type=float, description='variance of the visual noise (default = 2)', optional=True, default=2
            )
        })
        return params

    def __init__(self, *args, **kwargs):
        super().__init__(self._vif_diff, *args, **kwargs)
        self.meta['target'] = 'higher-better'
        self.meta['target_per_value'] = {'mean': 'higher-better', 'std': 'higher-worse', 'max_error': 'higher-worse'}
        if isinstance(convolve2d, UnsupportedPackage):
            convolve2d.raise_error(self.__provider__)

    def configure(self):
        super().configure()
        self.sigma_nsq = self.get_value_from_config('sigma_nsq')

    def _vif_diff(self, annotation_image, prediction_image):
        return np.mean(
            [self._vifp_single(annotation_image[:, :, i], prediction_image[:, :, i], self.sigma_nsq)
             for i in range(annotation_image.shape[2])]
        )

    @staticmethod
    def _vifp_single(gt, p, sigma_nsq):
        EPS = 1e-10
        num = 0.0
        den = 0.0
        for scale in range(1, 5):
            N = 2.0 ** (4 - scale + 1) + 1
            win = gaussian_filter(N, N / 5)

            if scale > 1:
                gt = convolve2d(gt, np.rot90(win, 2), mode='valid')[::2, ::2]
                p = convolve2d(p, np.rot90(win, 2), mode='valid')[::2, ::2]

            gt_sum_sq, p_sum_sq, gt_p_sum_mul = _get_sums(gt, p, win, mode='valid')
            sigmagt_sq, sigmap_sq, sigmagt_p = _get_sigmas(
                gt, p, win, mode='valid', sums=(gt_sum_sq, p_sum_sq, gt_p_sum_mul)
            )

            sigmagt_sq[sigmagt_sq < 0] = 0
            sigmap_sq[sigmap_sq < 0] = 0

            g = sigmagt_p / (sigmagt_sq + EPS)
            sv_sq = sigmap_sq - g * sigmagt_p

            g[sigmagt_sq < EPS] = 0
            sv_sq[sigmagt_sq < EPS] = sigmap_sq[sigmagt_sq < EPS]
            sigmagt_sq[sigmagt_sq < EPS] = 0

            g[sigmap_sq < EPS] = 0
            sv_sq[sigmap_sq < EPS] = 0

            sv_sq[g < 0] = sigmap_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= EPS] = EPS

            num += np.sum(np.log10(1.0 + (g ** 2.) * sigmagt_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1.0 + sigmagt_sq / sigma_nsq))

        return num / den


def gaussian_filter(ws, sigma):
    x, y = np.mgrid[-ws // 2 + 1:ws // 2 + 1, -ws // 2 + 1:ws // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    assert g.shape == (ws, ws)
    den = g.sum()
    if den != 0:
        g /= den
    return g


def _get_sums(gt, p, win, mode='same'):
    mu1 = convolve2d(gt, np.rot90(win, 2), mode=mode)
    mu2 = convolve2d(p, np.rot90(win, 2), mode=mode)
    return mu1 * mu1, mu2 * mu2, mu1 * mu2


def _get_sigmas(gt, p, win, mode='same', sums=None):
    if sums is not None:
        gt_sum_sq, p_sum_sq, gt_p_sum_mul = sums
    else:
        gt_sum_sq, p_sum_sq, gt_p_sum_mul = _get_sums(gt, p, win, mode)

    sigm_gt_sq = convolve2d(gt * gt, np.rot90(win, 2), mode) - gt_sum_sq
    sigm_p_sq = convolve2d(p * p, np.rot90(win, 2), mode) - p_sum_sq
    sigm_gt_p_mul = convolve2d(gt * p, np.rot90(win, 2), mode) - gt_p_sum_mul

    return sigm_gt_sq, sigm_p_sq, sigm_gt_p_mul


class LPIPS(BaseRegressionMetric):
    __provider__ = 'lpips'
    annotation_types = (
        SuperResolutionAnnotation, ImageProcessingAnnotation, ImageInpaintingAnnotation, StyleTransferAnnotation
    )
    prediction_types = (
        SuperResolutionPrediction, ImageProcessingPrediction, ImageInpaintingPrediction, StyleTransferPrediction
    )

    def __init__(self, *args, **kwargs):
        super().__init__(self.lpips_differ, *args, **kwargs)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'color_order': StringField(
                optional=True, choices=['BGR', 'RGB'], default='RGB',
                description="The field specified which color order BGR or RGB will be used during metric calculation."
            ),
            'normalized_images': BoolField(optional=True, default=False, description='images in [0, 1] range or not'),
            'net': StringField(
                optional=True, default='alex', choices=['alex', 'vgg', 'squeeze'],
                description='network for perceptual score'
            ),
            'distance_threshold': NumberField(
                optional=True, description='Distance threshold for getting image ratio greater distance',
                value_type=float, min_value=0, max_value=1
            )
        })

        return parameters

    def configure(self):
        super().configure()
        self.color_order = self.get_value_from_config('color_order')
        self.normalized_images = self.get_value_from_config('normalized_images')
        self.color_scale = 255 if not self.normalized_images else 1
        if isinstance(lpips, UnsupportedPackage):
            lpips.raise_error(self.__provider__)
        self.dist_threshold = self.get_value_from_config('distance_threshold')
        self.loss = self._create_loss()

    def lpips_differ(self, annotation_image, prediction_image):
        if self.color_order == 'BGR':
            annotation_image = annotation_image[:, :, ::-1]
            prediction_image = prediction_image[:, :, ::-1]
        gt_tensor = lpips.im2tensor(annotation_image, factor=self.color_scale / 2)
        pred_tensor = lpips.im2tensor(prediction_image, factor=self.color_scale / 2)
        return self.loss(gt_tensor, pred_tensor).item()

    def evaluate(self, annotations, predictions):
        results = super().evaluate(annotations, predictions)
        if self.dist_threshold:
            invalid_ratio = np.sum(np.array(self.magnitude) > self.dist_threshold) / len(self.magnitude)
            self.meta['names'].append('ratio_greater_{}'.format(self.dist_threshold))
            results += (invalid_ratio, )
        return results

    def _create_loss(self):
        import torch # pylint: disable=C0415
        import torchvision # pylint: disable=C0415
        net = self.get_value_from_config('net')
        model_weights = {
            'alex': ('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
                     'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
                     ),
            'squeeze': 'https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth',
            'vgg':  'https://download.pytorch.org/models/vgg16-397923af.pth'
        }
        model_classes = {
            'alex': torchvision.models.alexnet,
            'squeeze': torchvision.models.squeezenet1_1,
            'vgg': torchvision.models.vgg16
        }
        with tempfile.TemporaryDirectory(prefix='lpips_model', dir=Path.cwd()) as model_dir:
            weights = model_weights[net]
            if isinstance(weights, tuple):
                weights = weights[1] if torch.__version__ <= '1.6.0' else weights[0]
            preloaded_weights = torch.utils.model_zoo.load_url(
                weights, model_dir=model_dir, progress=False, map_location='cpu'
            )
        model = model_classes[net](pretrained=False)
        model.load_state_dict(preloaded_weights)
        feats = model.features
        loss = lpips.LPIPS(pnet_rand=True)
        for slice_id in range(1, loss.net.N_slices + 1):
            sl = getattr(loss.net, 'slice{}'.format(slice_id))
            for module_id in sl._modules: # pylint: disable=W0212
                sl._modules[module_id] = feats[int(module_id)] # pylint: disable=W0212
            setattr(loss.net, 'slice{}'.format(slice_id), sl)
        return loss
