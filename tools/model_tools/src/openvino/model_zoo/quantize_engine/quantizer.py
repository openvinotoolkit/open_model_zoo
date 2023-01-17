# Copyright (c) 2022-2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import sys
import tempfile
import yaml

from pathlib import Path
from typing import Set, List

from openvino.model_zoo import _common


DEFAULT_POT_CONFIG_BASE = {
    'compression': {
        'algorithms': [
            {
                'name': 'DefaultQuantization',
                'params': {
                    'preset': 'performance',
                    'stat_subset_size': 300,
                },
            },
        ],
    },
}


class Quantizer:
    def __init__(self, python: str, requested_precisions: str, output_dir: Path, model_dir: Path, pot: Path,
                 dataset_dir: Path, dry_run: bool):
        self.python = python
        self.pot_path = pot
        self.requested_precisions = requested_precisions
        self.output_dir = output_dir or model_dir
        self.model_dir = model_dir
        self.dry_run = dry_run
        self.dataset_dir = dataset_dir

    @property
    def dataset_dir(self) -> Set[str]:
        return self._dataset_dir

    @dataset_dir.setter
    def dataset_dir(self, value: str = None):
        # We can't mark it as required, because it's not required when --print_all is specified.
        # So we have to check it manually.
        if not value:
            sys.exit('--dataset_dir must be specified.')
        self._dataset_dir = value

    @property
    def requested_precisions(self) -> Set[str]:
        return self._requested_precisions

    @requested_precisions.setter
    def requested_precisions(self, value: Set[str] = None):
        unknown_precisions = value - _common.KNOWN_QUANTIZED_PRECISIONS.keys()
        if unknown_precisions:
            sys.exit('Unknown precisions specified: {}.'.format(', '.join(sorted(unknown_precisions))))

        self._requested_precisions = value

    def get_pot_cmd_prefix(self) -> List[str]:
        pot_path = self.pot_path
        if pot_path is None:
            pot_executable = shutil.which('pot')

            if pot_executable:
                pot_path = pot_executable
            else:
                try:
                    pot_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'tools/post_training_optimization_tool/main.py'
                except KeyError:
                    sys.exit('Unable to locate Post-Training Optimization Toolkit. '
                        + 'Use --pot or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')
        pot_cmd_prefix = [str(self.python), '--', str(pot_path)]
        return pot_cmd_prefix


    def quantize(self, reporter, model, precision, target_device, pot_cmd_prefix, pot_env, model_root=None) -> bool:
        input_precision = _common.KNOWN_QUANTIZED_PRECISIONS[precision]

        model_root = _common.MODEL_ROOT if model_root is None else model_root
        pot_config_base_path = model_root / model.subdirectory / 'quantization.yml'

        reporter.print_section_heading('{}Quantizing {} from {} to {}',
            '(DRY RUN) ' if self.dry_run else '', model.name, input_precision, precision)

        try:
            with pot_config_base_path.open('rb') as pot_config_base_file:
                pot_config_base = yaml.safe_load(pot_config_base_file)

        except FileNotFoundError:
            reporter.print('Unable to locate quantization.yml in {}, loading default POT config',
                           pot_config_base_path)
            pot_config_base = DEFAULT_POT_CONFIG_BASE

        accuracy_check_config = model_root/ model.subdirectory / 'accuracy-check.yml'
        if not accuracy_check_config.exists():
            reporter.error('Unable to locate accuracy-check.yml in {}', accuracy_check_config)

        pot_config_paths = {
            'engine': {
                'config': str(accuracy_check_config),
            },
            'model': {
                'model': str(self.model_dir / model.subdirectory / input_precision / (model.name + '.xml')),
                'weights': str(self.model_dir / model.subdirectory / input_precision / (model.name + '.bin')),
                'model_name': model.name,
            }
        }

        pot_config = {**pot_config_base, **pot_config_paths}

        if target_device:
            pot_config['compression']['target_device'] = target_device

        model_output_dir = self.output_dir / model.subdirectory / precision
        pot_config_path = model_output_dir / 'pot-config.json'

        reporter.print('Creating {}...', pot_config_path)
        pot_config_path.parent.mkdir(parents=True, exist_ok=True)
        with pot_config_path.open('w') as pot_config_file:
            json.dump(pot_config, pot_config_file, indent=4)
            pot_config_file.write('\n')

        pot_output_dir = model_output_dir / 'pot-output'
        pot_output_dir.mkdir(parents=True, exist_ok=True)

        pot_cmd = [*pot_cmd_prefix,
            '--config={}'.format(pot_config_path),
            '--direct-dump',
            '--output-dir={}'.format(pot_output_dir),
        ]

        reporter.print('Quantization command: {}', _common.command_string(pot_cmd))
        reporter.print('Quantization environment: {}',
            ' '.join('{}={}'.format(k, _common.quote_arg(v))
                for k, v in sorted(pot_env.items())))

        success = True

        if not self.dry_run:
            reporter.print(flush=True)

            success = reporter.job_context.subprocess(pot_cmd, env={**os.environ, **pot_env})

        reporter.print()
        if not success: return False

        if not self.dry_run:
            reporter.print('Moving quantized model to {}...', model_output_dir)
            for ext in ['.xml', '.bin']:
                (pot_output_dir / 'optimized' / (model.name + ext)).replace(
                    model_output_dir / (model.name + ext))
            reporter.print()

        return True

    def bulk_quantize(self, reporter, models, target_device, datasets_definition_fp=None, model_root=None) -> List[str]:
        failed_models = []

        with tempfile.TemporaryDirectory() as temp_dir:
            pot_cmd_prefix = self.get_pot_cmd_prefix()

            annotation_dir = Path(temp_dir) / 'annotations'
            annotation_dir.mkdir()

            datasets_definition_fp = _common.DATASET_DEFINITIONS if datasets_definition_fp is None \
                else datasets_definition_fp

            pot_env = {
                'ANNOTATIONS_DIR': str(annotation_dir),
                'DATA_DIR': str(self.dataset_dir),
                'DEFINITIONS_FILE': str(datasets_definition_fp),
            }

            for model in models:
                if not model.quantization_output_precisions:
                    reporter.print_section_heading('Skipping {} (quantization not supported)', model.name)
                    reporter.print()
                    continue

                model_precisions = self.requested_precisions & model.quantization_output_precisions

                if not model_precisions:
                    reporter.print_section_heading('Skipping {} (all precisions skipped)', model.name)
                    reporter.print()
                    continue

                pot_env.update({
                    'MODELS_DIR': str(self.model_dir / model.subdirectory)
                })

                for precision in sorted(model_precisions):
                    if not self.quantize(reporter, model, precision, target_device, pot_cmd_prefix, pot_env,
                                         model_root):
                        failed_models.append(model.name)
                        break
        return failed_models
