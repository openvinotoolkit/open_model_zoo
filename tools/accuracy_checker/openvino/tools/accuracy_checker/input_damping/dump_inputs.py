"""
Copyright (c) 2018-2022 Intel Corporation

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

from argparse import ArgumentParser
from functools import partial
from ..argparser import add_common_args, add_config_filtration_args, add_dataset_related_args
from ..config import ConfigReader
from ..utils import extract_image_representations, get_path
from ..evaluators.quantization_model_evaluator import create_dataset_attributes
from ..launcher import create_launcher, InputFeeder
from .input_dumper import InputDumper


def build_argparser():
    parser = ArgumentParser('dump_inputs')
    add_common_args(parser)
    add_config_filtration_args(parser)
    add_dataset_related_args(parser)
    parser.add_argument(
        '-o', '--output_dir', required=True, type=partial(get_path, is_directory=True),
        help='Output directory for dumping')
    parser.add_argument(
        '--output_format', required=False, default='npz',
        help='Format for output mapping. Supported: {}'.join(['bin', 'npz', 'npy', 'ark', 'pickle']))
    parser.add_argument('-n', '--num_samples', type=int, required=False)
    return parser


def dump_inputs(output_dir, data_loader, preprocessor, input_feeder=None, num_samples=None, dump_format='bin'):
    def _get_batch_input(batch_annotation, batch_input):
        batch_input = preprocessor.process(batch_input, batch_annotation)
        batch_data, batch_meta = extract_image_representations(batch_input)
        if input_feeder is None:
            return batch_data, batch_meta
        filled_inputs = input_feeder.fill_inputs(batch_input)
        return filled_inputs, batch_meta

    sample_dumper = InputDumper.provide(dump_format, {'type': dump_format}, output_dir)

    for batch_id, (_, batch_annotation, batch_input, _) in enumerate(data_loader):
        filled_inputs, batch_meta = _get_batch_input(batch_annotation, batch_input)
        sample_dumper(filled_inputs[0], batch_id)
        if num_samples is not None and batch_id == num_samples:
            break


def main():
    args = build_argparser().parse_args()
    config, mode = ConfigReader.merge(args)
    if mode != 'models':
        raise ValueError(f'Unsupported mode {mode}')
    for config_entry in config[mode]:
        data_loader, _, preprocessor, _ = create_dataset_attributes(config_entry['datasets'], '')
        try:
            launcher_config = config_entry['launchers'][0]
            launcher = create_launcher(launcher_config)
            launcher_inputs = launcher.inputs
            input_precision = launcher_config.get('_input_precision', [])
            input_layouts = launcher_config.get('_input_layout', '')
            input_feeder = InputFeeder(
                launcher.config.get('inputs', []), launcher_inputs, launcher.input_shape, launcher.fit_to_input,
                launcher.default_layout, launcher_config['framework'] == 'dummy',
                input_precision,
                input_layouts
            )
        except:
            input_feeder = None
        dump_inputs(args.output_dir, data_loader, preprocessor, input_feeder, args.num_samples, args.output_format)


if __name__ == '__main__':
    main()
