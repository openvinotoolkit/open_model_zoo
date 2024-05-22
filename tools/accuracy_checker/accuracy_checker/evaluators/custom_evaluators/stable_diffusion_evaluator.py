"""
Copyright (c) 2018-2024 Intel Corporation

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

from pathlib import Path
import inspect
from typing import Union, List, Optional, Dict
import numpy as np
import cv2
import PIL
from .base_custom_evaluator import BaseCustomEvaluator
from .base_models import BaseCascadeModel
from ...config import ConfigError
from ...utils import UnsupportedPackage, extract_image_representations, get_path
from ...representation import Text2ImageGenerationPrediction
from ...logging import print_info


try:
    from diffusers import DiffusionPipeline
except ImportError as err_diff:
    DiffusionPipeline = UnsupportedPackage("diffusers", err_diff)

try:
    from diffusers import LMSDiscreteScheduler
except ImportError as err_diff:
    LMSDiscreteScheduler = UnsupportedPackage("diffusers", err_diff)
try:
    import torch
except ImportError as err_torch:
    torch = UnsupportedPackage("torch", err_torch)

try:
    from transformers import AutoTokenizer
except ImportError as err_transformers:
    AutoTokenizer = UnsupportedPackage("transformers", err_transformers)


class PipelinedModel(BaseCascadeModel):
    def __init__(self, network_info, launcher, models_args, delayed_model_loading=False, config=None):
        super().__init__(network_info, launcher, delayed_model_loading)
        self.network_info = network_info
        self.launcher = launcher
        self.pipe = None
        self.config = config or {}
        self.seed = self.config.get("seed", 42)
        self.num_steps = self.config.get("num_inference_steps", 50)
        parts = ['text_encoder', "unet", 'vae_decoder', "vae_encoder"]
        network_info = self.fill_part_with_model(network_info, parts, models_args, False, delayed_model_loading)
        if not delayed_model_loading:
            self.create_pipeline(launcher)

    def create_pipeline(self, launcher, netowrk_info=None):
        tokenizer_config = self.config.get("tokenizer_id", "openai/clip-vit-large-patch14")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)
        scheduler_config = self.config.get("scheduler_config", {})
        scheduler = LMSDiscreteScheduler.from_config(scheduler_config)
        netowrk_info = netowrk_info or self.network_info
        self.pipe = OVStableDiffusionPipeline(
            launcher, tokenizer, scheduler, self.network_info,
            seed=self.seed, num_inference_steps=self.num_steps)

    def predict(self, identifiers, input_data, input_meta):
        preds = []
        for idx, prompt in zip(identifiers, input_data):
            pred = self.pipe(prompt, output_type="np")["sample"][0]
            preds.append(Text2ImageGenerationPrediction(idx, pred))
        return None, preds

    def release(self):
        del self.pipe

    def load_network(self, network_list, launcher):
        if self.pipe is None:
            self.create_pipeline(launcher, network_list)
            return
        self.pipe.reset_compiled_models()
        for network_dict in network_list:
            self.pipe.load_network(network_dict["model"], network_dict["name"])
        self.pipe.compile(launcher)

    def load_model(self, network_list, launcher):
        if self.pipe is None:
            self.create_pipeline(launcher, network_list)
            return
        self.pipe.reset_compiled_models()
        for network_dict in network_list:
            self.pipe.load_model(network_dict, launcher)
        self.pipe.compile(launcher)

    def get_network(self):
        models = self.pipe.get_models()
        model_list = []
        for model_part_name, model in models.items():
            model_list.append({"name": model_part_name, "model": model})
        return model_list


class StableDiffusionEvaluator(BaseCustomEvaluator):
    def __init__(self, dataset_config, launcher, model, orig_config):
        super().__init__(dataset_config, launcher, orig_config)
        self.model = model

    @classmethod
    def from_configs(cls, config, delayed_model_loading=False, orig_config=None):
        dataset_config, launcher, _ = cls.get_dataset_and_launcher_info(config)
        model = PipelinedModel(
                    config.get('network_info', {}), launcher, config.get('_models', []),
                    delayed_model_loading, config)
        return cls(dataset_config, launcher, model, orig_config)

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            batch_data, batch_meta = extract_image_representations(batch_inputs)
            batch_raw_prediction, batch_prediction = self.model.predict(
                batch_identifiers, batch_data, batch_meta
            )
            batch_annotation, batch_prediction = self.postprocessor.process_batch(
                batch_annotation, batch_prediction, batch_meta
            )
            metrics_result = self._get_metrics_result(batch_input_ids, batch_annotation, batch_prediction,
                                                      calculate_metrics)
            if output_callback:
                output_callback(batch_raw_prediction, metrics_result=metrics_result,
                                element_identifiers=batch_identifiers, dataset_indices=batch_input_ids)
            self._update_progress(progress_reporter, metric_config, batch_id, len(batch_prediction), csv_file)


def scale_fit_to_window(dst_width: int, dst_height: int, image_width: int, image_height: int):
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)


def preprocess(image: PIL.Image.Image, height, width):
    src_width, src_height = image.size
    dst_width, dst_height = scale_fit_to_window(
        width, height, src_width, src_height)
    image = np.array(image.resize((dst_width, dst_height),
                     resample=PIL.Image.Resampling.LANCZOS))[None, :]
    pad_width = width - dst_width
    pad_height = height - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    image = image.astype(np.float32) / 255.0
    image = 2.0 * image - 1.0
    image = image.transpose(0, 3, 1, 2)
    return image, {"padding": pad, "src_width": src_width, "src_height": src_height}


class OVStableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        launcher: "BaseLauncher", # noqa: F821
        tokenizer: "CLIPTokenizer", # noqa: F821
        scheduler: Union["LMSDiscreteScheduler"], # noqa: F821
        model_info: Dict,
        seed = None,
        num_inference_steps = 50
    ):
        super().__init__()
        self.scheduler = scheduler
        self.launcher = launcher
        self.tokenizer = tokenizer
        # self.height = height
        # self.width = width
        self.load_models(model_info, launcher, True)
        self.compile(launcher)
        if seed is not None:
            np.random.seed(seed)
        self.num_inference_steps = num_inference_steps

    def compile(self, launcher):
        unet_shapes = [inp.get_partial_shape() for inp in self.unet_model.inputs]
        if not unet_shapes[0][0].is_dynamic:
            unet_shapes = [inp.get_partial_shape() for inp in self.unet_model.inputs]
            unet_shapes[0][0] = -1
            unet_shapes[2][0] = -1
            self.unet_model.reshape(dict(zip(self.unet_model.inputs, unet_shapes)))
        self.unet = launcher.ie_core.compile_model(self.unet_model, launcher.device)
        self.text_encoder = launcher.ie_core.compile_model(self.text_encoder_model, launcher.device)
        self.vae_decoder = launcher.ie_core.compile_model(self.vae_decoder_model, launcher.device)
        if self.vae_encoder_model is not None:
            self.vae_encoder = launcher.ie_core.compile_model(self.vae_encoder_model, launcher.device)
        self._text_encoder_output = self.text_encoder.output(0)
        self._unet_output = self.unet.output(0)
        self._vae_d_output = self.vae_decoder.output(0)
        self._vae_e_output = self.vae_encoder.output(0) if self.vae_encoder is not None else None
        self.height = unet_shapes[0][2].get_length() * 8 if not unet_shapes[0][2].is_dynamic else 512
        self.width = unet_shapes[0][3].get_length() * 8 if not unet_shapes[0][3].is_dynamic else 512

    def get_models(self):
        model_dict = {"text_encoder": self.text_encoder_model, "unet": self.unet_model, "vae_decoder": self.vae_decoder}
        if self.vae_encoder_model is not None:
            model_dict["vae_encoder"] = self.vae_encoder_model
        return model_dict

    def reset_compiled_models(self):
        self._text_encoder_output = None
        self._unet_output = None
        self._vae_d_output = None
        self._vae_e_output = None
        self.unet = None
        self.text_encoder = None
        self.vae_decoder = None
        self.vae_encoder = None

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PIL.Image.Image = None,
        negative_prompt: Union[str, List[str]] = None,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        output_type: Optional[str] = "pil",
        strength: float = 1.0,
    ):
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get prompt text embeddings
        text_embeddings = self._encode_prompt(prompt, do_classifier_free_guidance=do_classifier_free_guidance,
                                              negative_prompt=negative_prompt)
        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(self.num_inference_steps, **extra_set_kwargs)
        timesteps, _ = self.get_timesteps(self.num_inference_steps, strength)
        latent_timestep = timesteps[:1]

        # get the initial random noise unless the user supplied it
        latents, meta = self.prepare_latents(image, latent_timestep)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # lcm_dreamshaper-v7 consist extra unet input
        is_extra_input = len(self.unet.inputs) == 4 and self.unet.inputs[3].any_name == 'timestep_cond'
        if is_extra_input:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            w = torch.tensor(guidance_scale).repeat(batch_size)
            w_embedding = self.get_w_embedding(w, embedding_dim=256)

        for t in self.progress_bar(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            inputs = [ latent_model_input, np.array(t, dtype=np.float32), text_embeddings ]
            if is_extra_input:
                inputs.append(w_embedding)

            # predict the noise residual
            noise_pred = self.unet(inputs)[self._unet_output]
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)["prev_sample"].numpy()
        # scale and decode the image latents with vae
        image = self.vae_decoder(latents)[self._vae_d_output]

        image = self.postprocess_image(image, meta, output_type)
        return {"sample": image}

    def _encode_prompt(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
            negative_prompt: Union[str, List[str]] = None
        ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(
            text_input_ids)[self._text_encoder_output]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(
                text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(
                text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(self, image: PIL.Image.Image = None, latent_timestep: "torch.Tensor" = None):
        latents_shape = (1, 4, self.height // 8, self.width // 8)
        noise = np.random.randn(*latents_shape).astype(np.float32)
        if image is None:
            # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
            if isinstance(self.scheduler, LMSDiscreteScheduler):
                noise = noise * self.scheduler.sigmas[0].numpy()
            return noise, {}
        input_image, meta = preprocess(image, self.width, self.height)
        moments = self.vae_encoder(input_image)[self._vae_e_output]
        mean, logvar = np.split(moments, 2, axis=1)
        std = np.exp(logvar * 0.5)
        latents = (mean + std * np.random.randn(*mean.shape)) * 0.18215
        latents = self.scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
        return latents, meta

    def postprocess_image(self, image: np.ndarray, meta: Dict, output_type: str = "np"):
        if "padding" in meta:
            pad = meta["padding"]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [img.resize((orig_width, orig_height),
                                    PIL.Image.Resampling.LANCZOS) for img in image]
        else:
            if "src_height" in meta:
                orig_height, orig_width = meta["src_height"], meta["src_width"]
                image = [cv2.resize(img, (orig_width, orig_width)) for img in image]
        return image

    def get_timesteps(self, num_inference_steps: int, strength: float):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

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

    def load_network(self, model, model_name):
        setattr(self, "{}_model".format(model_name), model)

    def load_model(self, network_info, launcher):
        model, weights = self.automatic_model_search(network_info)
        if weights:
            network = launcher.read_network(str(model), str(weights))
        else:
            network = launcher.read_network(str(model), None)
        self.load_network(network, network_info["name"])

    @staticmethod
    def automatic_model_search(network_info):
        model = Path(network_info['model'])
        model_name = network_info["name"]
        if model.is_dir():
            is_blob = network_info.get('_model_is_blob')
            if is_blob:
                model_list = list(model.glob('*{}.blob'.format(model_name)))
                if not model_list:
                    model_list = list(model.glob('*.blob'))
            else:
                model_list = list(model.glob('*{}*.xml'.format(model_name)))
                blob_list = list(model.glob('*{}*.blob'.format(model_name)))
                onnx_list = list(model.glob('*{}*.onnx'.format(model_name)))
                if not model_list and not blob_list and not onnx_list:
                    model_list = list(model.glob('*.xml'))
                    blob_list = list(model.glob('*.blob'))
                    onnx_list = list(model.glob('*.onnx'))
                if not model_list:
                    model_list = blob_list if blob_list else onnx_list
            if not model_list:
                raise ConfigError('Suitable model for {} not found'.format(model_name))
            if len(model_list) > 1:
                raise ConfigError('Several suitable models for {} found'.format(model_name))
            model = model_list[0]
        accepted_suffixes = ['.xml', '.onnx']
        if model.suffix not in accepted_suffixes:
            raise ConfigError('Models with following suffixes are allowed: {}'.format(accepted_suffixes))
        print_info('{} - Found model: {}'.format(model_name, model))
        if model.suffix in ['.blob', '.onnx']:
            return model, None
        weights = get_path(network_info.get('weights', model.parent / model.name.replace('xml', 'bin')))
        accepted_weights_suffixes = ['.bin']
        if weights.suffix not in accepted_weights_suffixes:
            raise ConfigError('Weights with following suffixes are allowed: {}'.format(accepted_weights_suffixes))
        print_info('{} - Found weights: {}'.format(model_name, weights))
        return model, weights

    def print_input_output_info(self):
        model_parts = ("text_encoder", "unet", "vae_decoder", "vae_encoder")
        for part in model_parts:
            part_model_id = "{}_model".format(part)
            model = getattr(self, part_model_id, None)
            if model is not None:
                self.launcher.print_input_output_info(model, part)

    @staticmethod
    def get_w_embedding(w, embedding_dim=512, dtype=torch.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb
