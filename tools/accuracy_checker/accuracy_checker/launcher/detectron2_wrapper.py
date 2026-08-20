"""
Copyright (c) 2026 Intel Corporation

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
import re
import urllib.request
from ..utils import UnsupportedPackage
from .pytorch_launcher import append_to_path, CHECKPOINT_URL_REGEX

try:
    import torch
    from torch import nn
except ImportError as torch_error:
    torch = UnsupportedPackage('torch', torch_error.msg)
    nn = None


class Detectron2Wrapper(nn.Module if nn else object):
    """
    Wraps a Detectron2 GeneralizedRCNN model to adapt it for accuracy_checker.

    Detectron2 expects: List[Dict[str, Tensor]] with 'image' key
    Accuracy_checker provides: Batch of tensors with shape [B, C, H, W]

    This wrapper:
    1. Takes batched tensor input [B, C, H, W]
    2. Normalizes layout to NCHW/CHW if needed
    3. Converts to detectron2 format: [{"image": tensor}, ...]
    4. Calls the detectron2 model
    5. Extracts predictions from Instances objects
    """

    @staticmethod
    def _raise_if_torch_unavailable():
        if nn is None:
            torch.raise_error(Detectron2Wrapper.__name__)

    @staticmethod
    def load_prebuilt_checkpoint(torch_launcher, checkpoint, model_cls_name, python_path):
        # Detectron2 checkpoints can contain a fully constructed model object;
        # load it directly to avoid calling the GeneralizedRCNN constructor.
        Detectron2Wrapper._raise_if_torch_unavailable()
        with append_to_path(python_path):
            if isinstance(checkpoint, str) and re.match(CHECKPOINT_URL_REGEX, checkpoint):
                checkpoint = urllib.request.urlretrieve(checkpoint)[0]  # nosec B310

            checkpoint_obj = torch.load(
                checkpoint,
                map_location=None if torch_launcher.cuda else torch.device('cpu'),
                weights_only=False
            )

            if isinstance(checkpoint_obj, nn.Module):
                return torch_launcher.prepare_module(checkpoint_obj, model_cls_name)

        return None

    @staticmethod
    def prepare_module(torch_launcher, module, model_class):
        Detectron2Wrapper._raise_if_torch_unavailable()
        wrapped_module = Detectron2Wrapper(module)
        wrapped_module.model.to('cuda' if torch_launcher.cuda else 'cpu')
        wrapped_module.model.eval()
        if torch_launcher.use_torch_compile:
            if hasattr(model_class, 'compile'):
                wrapped_module.model.compile()
            wrapped_module.model = torch.compile(
                wrapped_module.model,
                **torch_launcher.compile_kwargs
            )
        return wrapped_module

    def __init__(self, model):
        self._raise_if_torch_unavailable()
        super().__init__()
        self.model = model

    @staticmethod
    def _normalize_input_layout(batched_inputs):
        Detectron2Wrapper._raise_if_torch_unavailable()
        if isinstance(batched_inputs, torch.Tensor):
            if batched_inputs.dim() == 4 and batched_inputs.shape[1] != 3:
                if batched_inputs.shape[-1] == 3:
                    batched_inputs = batched_inputs.permute(0, 3, 1, 2).contiguous()
                elif batched_inputs.shape[2] == 3:
                    batched_inputs = batched_inputs.permute(0, 2, 1, 3).contiguous()
            elif batched_inputs.dim() == 3 and batched_inputs.shape[0] != 3:
                if batched_inputs.shape[-1] == 3:
                    batched_inputs = batched_inputs.permute(2, 0, 1).contiguous()
                elif batched_inputs.shape[1] == 3:
                    batched_inputs = batched_inputs.permute(1, 0, 2).contiguous()
        return batched_inputs

    @staticmethod
    def _to_detectron_inputs(batched_inputs):
        if isinstance(batched_inputs, (list, tuple)) and len(batched_inputs) > 0:
            if isinstance(batched_inputs[0], dict):
                return batched_inputs
            return [{"image": image} for image in batched_inputs]
        if batched_inputs.dim() == 4:
            return [{"image": image} for image in batched_inputs]
        if batched_inputs.dim() == 3:
            return [{"image": batched_inputs}]
        raise ValueError(f"Unexpected input shape: {batched_inputs.shape}")

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: Tensor of shape [B, C, H, W] or [C, H, W]

        Returns:
            Dict with extracted predictions or list of dicts if batch > 1
        """
        self._raise_if_torch_unavailable()
        batched_inputs = self._normalize_input_layout(batched_inputs)
        detectron_inputs = self._to_detectron_inputs(batched_inputs)

        # Call detectron2 model
        with torch.no_grad():
            outputs = self.model(detectron_inputs)

        # Extract instances from outputs
        # outputs is List[Dict[str, Instances]]
        instances_list = [out.get("instances") for out in outputs]

        # Convert Instances to dict of tensors for accuracy_checker
        results = []
        for instances in instances_list:
            pred_masks = torch.tensor([])
            if hasattr(instances, "pred_masks") and instances.pred_masks is not None:
                pred_masks = instances.pred_masks.float()

            result = {
                "pred_boxes": instances.pred_boxes.tensor if hasattr(instances, "pred_boxes") else torch.tensor([]),
                "pred_classes": instances.pred_classes if hasattr(instances, "pred_classes") else torch.tensor([]),
                "pred_masks": pred_masks,
                "scores": instances.scores if hasattr(instances, "scores") else torch.tensor([]),
            }
            results.append(result)

        # Return single dict or list of dicts based on batch size
        if len(results) == 1:
            return results[0]
        return results
