import os
from collections import OrderedDict

from ..config import NumberField, StringField, BoolField
from ..config import PathField, StringField, NumberField, BoolField, ConfigError
from .launcher import Launcher

import numpy as np

class DGLLauncher(Launcher):
    __provider__ = 'dgl'

    def __init__(self, config_entry: dict, *args, **kwargs):
        super().__init__(config_entry, *args, **kwargs)
        try:
            import dgl  # pylint: disable=C0415
            self._dgl = dgl
        except ImportError as import_error:
            raise ValueError(
                "DGL isn't installed. Please, install it before using. \n{}".format(
                    import_error.msg
                )
            )

        # self.validate_config(config_entry)

        # self._device = self.get_value_from_config("device")
        # self._vm_executor = self.get_value_from_config("vm")
        # self._batch = self.get_value_from_config("batch")

        # self._get_device()

        # self._generate_inputs()

        # self._module = self._load_module(config_entry["model"])

        # self._generate_outputs()

    @classmethod
    def parameters(cls):
        """Добавляем доп параметры для запуска
        """
        parameters = super().parameters()
        parameters.update({
            'model': PathField(description="Path to model.", file_or_directory=True)
        })
        return parameters
    
    def release(self):
        """
        Releases launcher.
        """
        pass