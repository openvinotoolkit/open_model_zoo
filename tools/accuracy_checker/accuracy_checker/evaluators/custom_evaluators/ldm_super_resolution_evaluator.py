"""
Copyright (c) 2024 Intel Corporation

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

import inspect
from typing import Union, List, Optional
import numpy as np
import PIL
from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseCascadeModel
from ...utils import UnsupportedPackage, extract_image_representations
from ...representation import SuperResolutionPrediction

try:
    from diffusers import DiffusionPipeline
except ImportError as err_diff:
    DiffusionPipeline = UnsupportedPackage("diffusers", err_diff)

try:
    from diffusers import LMSDiscreteScheduler
except ImportError as err_diff:
    LMSDiscreteScheduler = UnsupportedPackage("diffusers", err_diff)

try:
    from diffusers import DDIMScheduler
except ImportError as err_diff:
    DDIMScheduler = UnsupportedPackage("diffusers", err_diff)

try:
    from diffusers.utils import torch_utils
except ImportError as err_diff:
    torch_utils = UnsupportedPackage("diffusers.utils", err_diff)

try:
    import torch
except ImportError as err_torch:
    torch = UnsupportedPackage("torch", err_torch)


class PipelinedModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, delayed_model_loading=False, config=None):
        super().__init__(network_info, launcher, delayed_model_loading)
        self.network_info = network_info
        self.launcher = launcher
        self.pipe = None
        self.config = config or {}
        self.seed = self.config.get("seed", 42)
        self.num_steps = self.config.get("num_inference_steps", 100)
        parts = network_info.keys()
        network_info = self.fill_part_with_model(
            network_info, parts, models_args, False, delayed_model_loading
        )
        if not delayed_model_loading:
            self.create_pipeline(launcher)

    def create_pipeline(self, launcher, netowrk_info=None):
        netowrk_info = netowrk_info or self.network_info
        scheduler_config = self.config.get("scheduler_config", {})
        scheduler = LMSDiscreteScheduler.from_config(scheduler_config)

        self.load_models(netowrk_info, launcher, True)
        unet = launcher.ie_core.compile_model(self.unet_model, launcher.device)
        vqvae = launcher.ie_core.compile_model(self.vqvae_model, launcher.device)

        self.pipe = OVLdmSuperResolutionPipeline(
            launcher, scheduler, unet, vqvae,
            seed=self.seed, num_inference_steps=self.num_steps
        )

    def release(self):
        del self.pipe

    def load_models(self, model_info, launcher, log=False):
        if isinstance(model_info, dict):
            for model_name, model_dict in model_info.items():
                model_dict["name"] = model_name
                self.load_model(model_dict, launcher)
        else:
            for model_dict in model_info:
                self.load_model(model_dict, launcher)

        if log:
            self.print_input_output_info()

    def load_model(self, network_list, launcher):
        model, weights = self.automatic_model_search(network_list)
        if weights:
            network = launcher.read_network(str(model), str(weights))
        else:
            network = launcher.read_network(str(model), None)
        setattr(self, f"{network_list['name']}_model", network)

    def print_input_output_info(self):
        model_parts = ("unet", "vqvae")
        for part in model_parts:
            part_model_id = f"{part}_model"
            model = getattr(self, part_model_id, None)
            if model is not None:
                self.launcher.print_input_output_info(model, part)

    def predict(self, identifiers, input_data, input_meta):
        preds = []
        for idx, image in zip(identifiers, input_data):
            pred = self.pipe(image, eta=1, output_type="np")["hr_sample"][0]
            preds.append(SuperResolutionPrediction(idx, pred))
        return preds


class LdmSuperResolutionEvaluator(BaseCustomEvaluator):
    def __init__(self, model, dataset_config, launcher, preprocessor, postprocessor, orig_config):
        super().__init__(dataset_config, launcher, orig_config, preprocessor, postprocessor)
        self.model = model

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, preprocessor, postprocessor = (
            BaseCustomEvaluator.get_evaluator_init_info(
                config, delayed_annotation_loading=False
            )
        )

        model = PipelinedModel(
            config.get('network_info', {}), launcher, config.get('_models', []),
            delayed_model_loading, config
        )

        return cls(
            model, dataset_config, launcher, preprocessor, postprocessor, orig_config
        )

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_input, batch_identifiers) in enumerate(self.dataset):
            batch_input = self.preprocessor.process(batch_input, batch_annotation)

            batch_data, batch_meta = extract_image_representations(batch_input)
            batch_prediction = self.model.predict(
                batch_identifiers, batch_data, batch_meta
            )
            batch_annotation, batch_prediction = self.postprocessor.process_batch(
                batch_annotation, batch_prediction, batch_meta
            )

            metrics_result = self._get_metrics_result(
                batch_input_ids, batch_annotation, batch_prediction, calculate_metrics
            )

            if output_callback:
                output_callback(
                    batch_raw_prediction=None, metrics_result=metrics_result,
                    element_identifiers=batch_identifiers, dataset_indices=batch_input_ids
                )
            self._update_progress(
                progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file
            )


class OVLdmSuperResolutionPipeline(DiffusionPipeline):
    def __init__(
        self,
        launcher: "BaseLauncher",  # noqa: F821
        scheduler: Union[DDIMScheduler, LMSDiscreteScheduler],
        unet,
        vqvae,
        seed=None,
        num_inference_steps=100
    ):
        super().__init__()
        self.launcher = launcher
        self.scheduler = scheduler
        self.unet = unet
        self.vqvae = vqvae
        self._unet_output = self.unet.output(0)
        self._vqvae_output = self.vqvae.output(0)
        if seed is not None:
            torch.manual_seed(seed)
        self.num_inference_steps = num_inference_steps

    def __call__(
        self,
        image: Union[torch.Tensor, np.ndarray, PIL.Image.Image] = None,
        batch_size: Optional[int] = 1,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        output_type: Optional[str] = "pil",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True
    ):

        batch_size, image = self.preprocess_image(image)
        height, width = image.shape[-2:]

        # in_channels should be 6: 3 for latents, 3 for low resolution image
        latents_shape = (batch_size, 3, height, width)
        latents = torch_utils.randn_tensor(latents_shape, generator=generator)
        # set timesteps and move to the correct device
        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        latents = latents.numpy()
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        for t in self.progress_bar(timesteps_tensor):
            # concat latents and low resolution image in the channel dimension.
            latents_input = np.concatenate([latents, image], axis=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            # predict the noise residual
            noise_pred = self.unet([latents_input, t])[self._unet_output]
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents)
            )["prev_sample"].numpy()

        # decode the image latents with the VQVAE
        image = self.vqvae(latents)[self._vqvae_output]

        image = self.postprocess_image(image, std=255, mean=0)

        return {"hr_sample": image}

    @staticmethod
    def preprocess_image(image):
        if isinstance(image, PIL.Image.Image):
            w, h = image.size
            w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
            image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
            image = np.array(image)
            batch_size = 1
        elif isinstance(image, torch.Tensor):
            image = np.array(image)
            batch_size = image.shape[0]
        elif isinstance(image, np.ndarray):
            batch_size = 1
        else:
            raise ValueError(
                f"`image` has to be of type `PIL.Image.Image` or `np.ndarray` or `torch.Tensor` but is {type(image)}"
            )

        image = image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        timage = 2.0 * image - 1.0

        return batch_size, timage

    @staticmethod
    def postprocess_image(image: np.ndarray, std=255, mean=0):
        image = image / 2 + 0.5
        image = image.transpose(0, 2, 3, 1)
        image *= np.array(std, dtype=image.dtype)
        image += np.array(mean, dtype=image.dtype)

        image = np.clip(image, 0., 255.)
        image = image.astype(np.uint8)
        return image
