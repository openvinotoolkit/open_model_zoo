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

from collections import namedtuple
from copy import deepcopy
from csv import DictWriter
import numpy as np

from .utils import Color, color_format, check_file_existence
from .dependency import ClassProvider
from .logging import print_info

EvaluationResult = namedtuple(
    'EvaluationResult', [
        'evaluated_value', 'reference_value', 'name', 'metric_type', 'abs_threshold', 'rel_threshold', 'meta',
        'profiling_file'
    ]
)


class BasePresenter(ClassProvider):
    __provider_type__ = "presenter"

    def write_result(self, evaluation_result, ignore_results_formatting=False, ignore_metric_reference=False):
        raise NotImplementedError

    def extract_result(self, evaluation_result, names_from_refs=False):
        raise NotImplementedError


class ScalarPrintPresenter(BasePresenter):
    __provider__ = "print_scalar"

    def write_result(self, evaluation_result: EvaluationResult, ignore_results_formatting=False,
                     ignore_metric_reference=False):
        value, reference, name, _, abs_threshold, rel_threshold, meta, _ = evaluation_result
        value = '' if not np.isscalar(value) and (value is None or None in value) else float(np.mean(value))
        postfix, scale, result_format = get_result_format_parameters(meta, ignore_results_formatting)
        difference = None
        if reference and not ignore_metric_reference:
            difference = compare_with_ref(reference, value, name)
        write_scalar_result(
            value, name, abs_threshold, rel_threshold, difference,
            postfix=postfix, scale=scale, result_format=result_format
        )

    def extract_result(self, evaluation_result, names_from_refs=False):
        value, ref, name, metric_type, abs_threshold, rel_threshold, meta, profiling_file = evaluation_result
        if isinstance(ref, dict):
            ref = ref.get(name)
        result_dict = {
            'name': name,
            'value': '' if not np.isscalar(value) and (value is None or None in value) else float(np.mean(value)),
            'type': metric_type,
            'ref': ref or '',
            'abs_threshold': abs_threshold or 0,
            'ref_threshold': rel_threshold or 0,
            'profiling_file': profiling_file
        }
        return result_dict, meta


class VectorPrintPresenter(BasePresenter):
    __provider__ = "print_vector"

    def write_result(self, evaluation_result: EvaluationResult, ignore_results_formatting=False,
                     ignore_metric_reference=False):
        value, reference, name, _, abs_threshold, rel_threshold, meta, _ = evaluation_result
        if abs_threshold:
            abs_threshold = float(abs_threshold)
        if rel_threshold:
            rel_threshold = float(rel_threshold)

        value_names = meta.get('names')
        postfix, scale, result_format = get_result_format_parameters(meta, ignore_results_formatting)
        if np.isscalar(value) or np.size(value) == 1:
            if not np.isscalar(value):
                if np.ndim(value) == 0:
                    value = value.tolist()
                else:
                    value = value[0]
            difference = None
            value_name = value_names[0] if value_names else None
            if reference and not ignore_metric_reference:
                difference = compare_with_ref(reference, value, value_name)
            write_scalar_result(
                value, name, abs_threshold, rel_threshold, difference,
                value_name=value_name,
                postfix=postfix[0] if not np.isscalar(postfix) else postfix,
                scale=scale[0] if not np.isscalar(scale) else scale,
                result_format=result_format
            )
            return

        for index, res in enumerate(value):
            difference = None
            value_scale = scale[index] if not np.isscalar(scale) else scale
            value_name = value_names[index] if value_names else None

            if reference and not ignore_metric_reference and isinstance(reference, dict):
                difference = compare_with_ref(reference, res, value_name)
            write_scalar_result(
                res, name, abs_threshold, rel_threshold, difference,
                value_name=value_name,
                postfix=postfix[index] if not np.isscalar(postfix) else postfix,
                scale=value_scale,
                result_format=result_format
            )

        if len(value) > 1 and meta.get('calculate_mean', True):
            mean_value = '' if not np.isscalar(value) and (value is None or None in value) else float(np.mean(value))
            value_scale = scale[0] if not np.isscalar(scale) else scale
            difference = None
            if reference and not ignore_metric_reference:
                difference = compare_with_ref(reference, mean_value, 'mean')
            write_scalar_result(
                mean_value, name, abs_threshold, rel_threshold, difference, value_name='mean',
                postfix=postfix[-1] if not np.isscalar(postfix) else postfix, scale=value_scale,
                result_format=result_format
            )

    def extract_result(self, evaluation_result, names_from_refs=False):
        value, reference, name, metric_type, abs_threshold, rel_threshold, meta, profiling_file = evaluation_result
        len_value = len(value) if not np.isscalar(value) and np.ndim(value) > 0 else 1
        value_names_orig = meta.get('names', list(range(0, len_value)))
        if names_from_refs and isinstance(reference, dict):
            calculate_mean = meta.get('calculate_mean', True)
            value_names_orig = (
                [name for name in reference if name != 'mean'] if calculate_mean else list(reference.keys())
            )
        value_names = ['{}@{}'.format(name, value_name) for value_name in value_names_orig]
        if np.isscalar(value) or np.size(value) <= 1:
            value_name = value_names[0] if value_names and 'names' in meta else name
            value_name_orig = value_names_orig[0] if value_name != name else name
            if isinstance(reference, dict):
                ref = reference.get(value_name_orig, '')
            else:
                ref = reference or ''
            if not np.isscalar(value):
                if np.ndim(value) == 0:
                    value = value.tolist()
                else:
                    value = value[0]
            result_dict = {
                'name': value_name,
                'value': value,
                'type': metric_type,
                'ref': ref,
                'abs_threshold': abs_threshold or 0,
                'rel_threshold': rel_threshold or 0,
                'profiling_file': profiling_file
            }
            return result_dict, meta

        if meta.get('calculate_mean', True):
            value_names_orig.append('mean')
            value_names.append('{}@mean'.format(name))
            mean_value = '' if not np.isscalar(value) and (value is None or None in value) else float(np.mean(value))
            value = np.append(value, mean_value)
        per_value_meta = []
        target_per_value = meta.pop('target_per_value', {})
        target = meta.pop('target', 'higher-better')
        for orig_name in value_names_orig:
            target_for_value = target_per_value.get(orig_name, target)
            meta_for_value = deepcopy(meta)
            meta_for_value['target'] = target_for_value
            meta_for_value['class_name'] = orig_name
            per_value_meta.append(meta_for_value)
        results = []
        for idx, value_item in enumerate(value):
            orig_name = value_names_orig[idx]
            ref = ''
            if isinstance(reference, dict):
                ref = reference.get(orig_name, '')
            results.append(
                {
                    'name': value_names[idx],
                    'value': value_item,
                    'type': metric_type,
                    'ref': ref,
                    'abs_threshold': 0,
                    'rel_threshold': 0
                }
            )
        return results, per_value_meta


def write_scalar_result(
        res_value, name, abs_threshold=None, rel_threshold=None, diff_with_ref=None, value_name=None,
        postfix='%', scale=100, result_format='{:.2f}'
):
    display_name = "{}@{}".format(name, value_name) if value_name else name
    display_result = result_format.format(res_value * scale)
    message = '{}: {}{}'.format(display_name, display_result, postfix)

    if diff_with_ref and (diff_with_ref[0] or diff_with_ref[1]):
        abs_threshold = abs_threshold or 0
        rel_threshold = rel_threshold or 0
        if abs_threshold <= diff_with_ref[0] or rel_threshold <= diff_with_ref[1]:
            fail_message = "[FAILED:  abs error = {:.4} | relative error = {:.4}]".format(
                diff_with_ref[0] * scale, diff_with_ref[1]
            )
            message = "{} {}".format(message, color_format(fail_message, Color.FAILED))
        else:
            message = "{} {}".format(message, color_format("[OK]", Color.PASSED))

    print_info(message)


def compare_with_ref(reference, res_value, name=None):
    if isinstance(reference, dict):
        if name is None:
            ref = next(iter(reference.values()))
        else:
            ref = reference.get(name)
    else:
        ref = reference
    ref = ref if ref is None else float(ref)
    if ref is None:
        return None
    return abs(ref - res_value), abs(ref - res_value) / ref


def get_result_format_parameters(meta, use_default_formatting):
    postfix = ' '
    scale = 1
    result_format = '{}'
    if not use_default_formatting:
        postfix = meta.get('postfix', '%')
        scale = meta.get('scale', 100)
        result_format = meta.get('data_format', '{:.2f}')

    return postfix, scale, result_format


def generate_csv_report(processing_info, metric_results, dataset_size, metrics_meta):
    field_names = [
        'model', 'launcher', 'device', 'dataset',
        'tags', 'metric_name', 'metric_type', 'metric_value', 'metric_target', 'metric_scale', 'metric_postfix',
        'dataset_size', 'ref', 'abs_threshold', 'rel_threshold', 'profiling_file']
    model, launcher, device, tags, dataset = processing_info
    main_info = {
        'model': model,
        'launcher': launcher,
        'device': device.upper(),
        'tags': ' '.join(tags) if tags else '',
        'dataset': dataset,
        'dataset_size': dataset_size
    }
    rows = []
    for metric_result, metric_meta in zip(metric_results, metrics_meta):
        rows.append({
            **main_info,
            'metric_name': metric_result['name'],
            'metric_type': metric_result['type'],
            'metric_value': metric_result['value'],
            'metric_target': metric_meta.get('target', 'higher-better'),
            'metric_scale': metric_meta.get('scale', 100),
            'metric_postfix': metric_meta.get('postfix', '%'),
            'ref': metric_result.get('ref', ''),
            'abs_threshold': metric_result.get('abs_threshold', 0),
            'rel_threshold': metric_result.get('rel_threshold', 0),
            'profiling_file': metric_result.get('profiling_file', '')
        })
    return field_names, rows


def write_csv_result(csv_file, processing_info, metric_results, dataset_size, metrics_meta):
    new_file = not check_file_existence(csv_file)
    field_names, rows = generate_csv_report(processing_info, metric_results, dataset_size, metrics_meta)
    with open(csv_file, 'a+', newline='', encoding='utf-8') as f:
        writer = DictWriter(f, fieldnames=field_names)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
