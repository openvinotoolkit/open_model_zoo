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

import pickle as pkl
import json
from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn
from ..representation import CharacterRecognitionAnnotation
from ..utils import read_txt, check_file_existence
from ..config import PathField


def read_vocab(vocab_path):
    """Reads vocab file from disk as .pkl or .json

    Args:
        vocab_path (str): path to vocab file

    Raises:
        ValueError: If wrong extension of the file

    Returns:
        Vocab: Vocab object with sign2id and id2sign dictinaries
    """
    if vocab_path.suffix == '.pkl':
        with open(vocab_path, "rb") as f:
            vocab_dict = pkl.load(f)
    elif vocab_path.suffix == '.json':
        with open(vocab_path, "r") as f:
            vocab_dict = json.load(f)
            id2sign = {int(k): v for k, v in vocab_dict['id2sign'].items()}
            vocab_dict['id2sign'] = id2sign
    else:
        raise ValueError("Wrong extension of the vocab file")
    return vocab_dict["id2sign"]


class Im2latexDatasetConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'im2latex_formula_recognition'
    annotation_types = (CharacterRecognitionAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update(
            {
                'images_dir': PathField(
                    is_directory=True, optional=False,
                    description='path to input images'
                ),
                'formula_file': PathField(
                    optional=True,
                    description='path to file containing one formula per line'
                ),
                'split_file': PathField(
                    optional=True,
                    description='path to split containing image_name\\tformula_idx'
                ),
                'vocab_file': PathField(
                    optional=True,
                    description='path to vocabulary'
                ),
            }
        )
        return configuration_parameters

    def configure(self):
        super().configure()
        self.images_dir = self.get_value_from_config('images_dir')
        self.formula_path = self.get_value_from_config('formula_file')
        self.split_path = self.get_value_from_config('split_file')
        self.vocab_path = self.get_value_from_config('vocab_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        """Reads data from disk and returns dataset in converted for AC format

        Args:
            check_content (bool, optional): Check if content is valid. Defaults to False.
            progress_callback (bool, optional): Display progress. Defaults to None.
            progress_interval (int, optional): Units to display progress. Defaults to 100 (percent).

        Returns:
            [type]: Converted dataset
        """
        annotations = []
        content_errors = None if not check_content else []
        split_file = read_txt(self.split_path)
        formulas_file = read_txt(self.formula_path)
        num_iterations = len(split_file)
        vocab = read_vocab(self.vocab_path)

        for line_id, line in enumerate(split_file):
            img_name, formula_idx = line.split('\t')
            gt_formula = formulas_file[int(formula_idx)]
            annotations.append(CharacterRecognitionAnnotation(img_name, gt_formula))
            if check_content:
                if not check_file_existence(self.images_dir / img_name):
                    content_errors.append('{}: does not exist'.format(img_name))
            if progress_callback is not None and line_id % progress_interval == 0:
                progress_callback(line_id / num_iterations * 100)

        meta = {'vocab': vocab}

        return ConverterReturn(annotations, meta, content_errors)
