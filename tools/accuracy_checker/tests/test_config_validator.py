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

from math import inf, nan
from pathlib import Path
from unittest.mock import ANY

import pytest
from accuracy_checker.config.config_validator import (
    ConfigError,
    ConfigValidator,
    DictField,
    ListField,
    NumberField,
    PathField,
    StringField,
    BaseField,
    BoolField
)
from accuracy_checker.evaluators import ModelEvaluator
from accuracy_checker.launcher import Launcher
from accuracy_checker.dataset import Dataset
from accuracy_checker.metrics import Metric
from accuracy_checker.postprocessor import Postprocessor
from accuracy_checker.preprocessor import Preprocessor
from accuracy_checker.data_readers import BaseReader
from accuracy_checker.annotation_converters import BaseFormatConverter
from accuracy_checker.adapters import Adapter
from accuracy_checker.utils import contains_all
from tests.common import mock_filesystem


class TestStringField:
    def test_expects_string(self):
        string_field = StringField()

        with pytest.raises(ConfigError):
            string_field.validate(b"foo")
        with pytest.raises(ConfigError):
            string_field.validate({})
        with pytest.raises(ConfigError):
            string_field.validate(42)

        string_field.validate("foo")

    def test_choices(self):
        string_field = StringField(choices=['foo', 'bar'])

        with pytest.raises(ConfigError):
            string_field.validate('baz')

        string_field.validate('bar')

    def test_case_sensitive(self):
        string_field = StringField(choices=['foo', 'bar'], case_sensitive=False)

        string_field.validate('foo')
        string_field.validate('FOO')

        string_field = StringField(choices=['foo', 'bar'], case_sensitive=True)

        string_field.validate('foo')
        with pytest.raises(ConfigError):
            string_field.validate('FOO')

    def test_regex(self):
        string_field = StringField(regex=r'foo\d*')

        string_field.validate('foo')
        string_field.validate('foo42')

        with pytest.raises(ConfigError):
            string_field.validate('baz')

    def test_custom_exception(self, mocker):
        stub = mocker.stub(name='custom_on_error')
        string_field = StringField(choices=['foo'], on_error=stub)

        with pytest.raises(ConfigError):
            string_field.validate('bar', 'foo')
        stub.assert_called_once_with('bar', 'foo', ANY)

    def test_custom_validator(self, mocker):
        stub = mocker.stub(name='custom_validator')
        string_field = StringField(choices=['foo'], additional_validator=stub)

        string_field.validate('foo', 'baz')
        stub.assert_called_once_with('foo', 'baz')


class TestNumberField:
    def test_expects_number(self):
        number_field = NumberField(value_type=float)

        number_field.validate(1.0)
        with pytest.raises(ConfigError):
            number_field.validate("foo")
        with pytest.raises(ConfigError):
            number_field.validate({})
        with pytest.raises(ConfigError):
            number_field.validate([])

        number_field = NumberField(value_type=int)
        number_field.validate(1)
        with pytest.raises(ConfigError):
            number_field.validate(1.0)

    def test_nans(self):
        number_field = NumberField(allow_nan=True)
        number_field.validate(nan)

        number_field = NumberField(allow_nan=False)
        with pytest.raises(ConfigError):
            number_field.validate(nan)

    def test_infinity(self):
        number_field = NumberField(allow_inf=True)
        number_field.validate(inf)

        number_field = NumberField(allow_inf=False)
        with pytest.raises(ConfigError):
            number_field.validate(inf)

    def test_ranges(self):
        number_field = NumberField(min_value=0, max_value=5)

        number_field.validate(0)
        number_field.validate(1)
        number_field.validate(2)

        with pytest.raises(ConfigError):
            number_field.validate(-1)
        with pytest.raises(ConfigError):
            number_field.validate(7)


class TestDictField:
    def test_expects_dict(self):
        dict_field = DictField()

        dict_field.validate({})
        with pytest.raises(ConfigError):
            dict_field.validate("foo")
        with pytest.raises(ConfigError):
            dict_field.validate(42)
        with pytest.raises(ConfigError):
            dict_field.validate([])

    def test_validates_keys(self):
        dict_field = DictField()
        dict_field.validate({'foo': 42, 1: 'bar'})

        dict_field = DictField(key_type=str)
        dict_field.validate({'foo': 42, 'bar': 'bar'})
        with pytest.raises(ConfigError):
            dict_field.validate({'foo': 42, 1: 'bar'})

        dict_field = DictField(key_type=StringField(choices=['foo', 'bar']))
        dict_field.validate({'foo': 42, 'bar': 42})
        with pytest.raises(ConfigError):
            dict_field.validate({'foo': 42, 1: 'bar'})
        with pytest.raises(ConfigError):
            dict_field.validate({'foo': 42, 'baz': 42})

    def test_validates_values(self):
        dict_field = DictField()
        dict_field.validate({'foo': 42, 1: 'bar'})

        dict_field = DictField(value_type=str)
        dict_field.validate({'foo': 'foo', 1: 'bar'})
        with pytest.raises(ConfigError):
            dict_field.validate({'foo': 42, 1: 2})

        dict_field = DictField(value_type=StringField(choices=['foo', 'bar']))
        dict_field.validate({1: 'foo', 'bar': 'bar'})
        with pytest.raises(ConfigError):
            dict_field.validate({1: 'foo', 2: 3})
        with pytest.raises(ConfigError):
            dict_field.validate({1: 'foo', 2: 'baz'})

    def test_converts_basic_types(self):
        dict_field = DictField(value_type=str)
        assert isinstance(dict_field.value_type, StringField)

        dict_field = DictField(value_type=int)
        assert isinstance(dict_field.value_type, NumberField)
        assert dict_field.value_type.type is not float

        dict_field = DictField(value_type=float)
        assert isinstance(dict_field.value_type, NumberField)
        assert dict_field.value_type.type is float

        dict_field = DictField(value_type=list)
        assert isinstance(dict_field.value_type, ListField)

        dict_field = DictField(value_type=dict)
        assert isinstance(dict_field.value_type, DictField)

        dict_field = DictField(value_type=Path)
        assert isinstance(dict_field.value_type, PathField)

    def test_empty(self):
        dict_field = DictField()
        dict_field.validate({})

        dict_field = DictField(allow_empty=False)
        with pytest.raises(ConfigError):
            dict_field.validate({})


class TestListField:
    def test_expects_list(self):
        list_field = ListField()

        list_field.validate([])
        with pytest.raises(ConfigError):
            list_field.validate("foo")
        with pytest.raises(ConfigError):
            list_field.validate(42)
        with pytest.raises(ConfigError):
            list_field.validate({})

    def test_validates_values(self):
        list_field = ListField()
        list_field.validate(['foo', 42])

        list_field = ListField(value_type=str)
        list_field.validate(['foo', 'bar'])
        with pytest.raises(ConfigError):
            list_field.validate(['foo', 42])

        list_field = ListField(value_type=StringField(choices=['foo', 'bar']))
        list_field.validate(['foo', 'bar'])
        with pytest.raises(ConfigError):
            list_field.validate(['foo', 42])
        with pytest.raises(ConfigError):
            list_field.validate(['foo', 'bar', 'baz'])

    def test_empty(self):
        list_field = ListField()
        list_field.validate([])

        list_field = ListField(allow_empty=False)
        with pytest.raises(ConfigError):
            list_field.validate([])


class TestPathField:
    @pytest.mark.usefixtures('mock_path_exists')
    def test_expects_path_like(self):
        path_field = PathField()
        path_field.validate('foo/bar')
        path_field.validate('/home/user')
        path_field.validate(Path('foo/bar'))

        with pytest.raises(ConfigError):
            path_field.validate(42)
        with pytest.raises(ConfigError):
            path_field.validate({})
        with pytest.raises(ConfigError):
            path_field.validate([])

    def test_path_is_checked(self):
        with mock_filesystem(['foo/bar']) as prefix:
            prefix_path = Path(prefix)
            file_field = PathField(is_directory=False)
            with pytest.raises(ConfigError):
                file_field.validate(prefix_path / 'foo')
            file_field.validate(prefix_path / 'foo' / 'bar')

            dir_field = PathField(is_directory=True)
            dir_field.validate(prefix_path / 'foo')

            with pytest.raises(ConfigError):
                dir_field.validate(prefix_path / 'foo' / 'bar')

    def test_path_not_checked(self):
        with mock_filesystem(['foo/bar']) as prefix:
            prefix_path = Path(prefix)
            file_field = PathField(is_directory=False, check_exists=False)
            file_field.validate(prefix_path / 'foo' / 'bar')


class TestConfigValidator:
    def test_compound(self):
        class SampleValidator(ConfigValidator):
            foo = StringField(choices=['foo'])
            bar = NumberField()

        sample_validator = SampleValidator('Sample')
        sample_validator.validate({'foo': 'foo', 'bar': 1})

        with pytest.raises(ConfigError):
            sample_validator.validate({'foo': 'foo'})
        with pytest.raises(ConfigError):
            sample_validator.validate({'foo': 'bar', 'bar': 1})

    def test_optional_fields(self):
        class SampleValidatorNoOptionals(ConfigValidator):
            foo = StringField(choices=['foo'])
            bar = NumberField(optional=False)

        sample_validator = SampleValidatorNoOptionals('Sample')
        sample_validator.validate({'foo': 'foo', 'bar': 1})
        with pytest.raises(ConfigError):
            sample_validator.validate({'foo': 'bar'})

        class SampleValidatorWithOptionals(ConfigValidator):
            foo = StringField(choices=['foo'])
            bar = NumberField(optional=True)

        sample_validator = SampleValidatorWithOptionals('Sample')
        sample_validator.validate({'foo': 'foo', 'bar': 1})
        sample_validator.validate({'foo': 'foo'})

    def test_extra_fields__warn_on_extra(self):
        class SampleValidatorWarnOnExtra(ConfigValidator):
            foo = StringField(choices=['foo'])

        sample_validator = SampleValidatorWarnOnExtra(
            'Sample', on_extra_argument=ConfigValidator.WARN_ON_EXTRA_ARGUMENT
        )

        with pytest.warns(UserWarning):
            sample_validator.validate({'foo': 'foo', 'bar': 'bar'})

    def test_extra_fields__error_on_extra(self):
        class SampleValidatorErrorOnExtra(ConfigValidator):
            foo = StringField(choices=['foo'])

        sample_validator = SampleValidatorErrorOnExtra(
            'Sample', on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

        with pytest.raises(ConfigError):
            sample_validator.validate({'foo': 'bar', 'bar': 'bar'})

    def test_extra_fields__ignore_extra(self):
        class SampleValidatorIgnoresExtra(ConfigValidator):
            foo = StringField(choices=['foo'])

        sample_validator = SampleValidatorIgnoresExtra(
            'Sample', on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

        sample_validator.validate({'foo': 'foo', 'bar': 'bar'})

    def test_custom_exception(self, mocker):
        class SampleValidator(ConfigValidator):
            foo = StringField(choices=['foo'])

        stub = mocker.stub(name='custom_on_error')
        sample_validator = SampleValidator('Sample', on_error=stub)
        with pytest.raises(ConfigError):
            sample_validator.validate({})
            stub.assert_called_once_with(ANY, 'Sample', ANY)

    def test_custom_validator(self, mocker):
        class SampleValidator(ConfigValidator):
            foo = StringField(choices=['foo'])

        stub = mocker.stub(name='custom_validator')
        sample_validator = SampleValidator('Sample', additional_validator=stub)
        entry = {'foo': 'foo'}
        sample_validator.validate(entry)
        stub.assert_called_once_with(entry, 'Sample')

    def test_nested(self):
        class InnerValidator(ConfigValidator):
            foo = StringField(choices=['foo'])

        class OuterValidator(ConfigValidator):
            bar = ListField(InnerValidator('Inner'))

        outer_validator = OuterValidator('Outer', on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)

        outer_validator.validate({'bar': [{'foo': 'foo'}, {'foo': 'foo'}]})

    def test_inheritance(self):
        class ParentValidator(ConfigValidator):
            foo = StringField(choices=['foo'])

        class DerivedValidator(ParentValidator):
            bar = StringField(choices=['bar'])

        derived_validator = DerivedValidator('Derived', on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT)
        derived_validator.validate({'foo': 'foo', 'bar': 'bar'})


class TestConfigValidationAPI:
    def test_empty_config(self):
        config_errors = ModelEvaluator.validate_config({'models': [{}]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.launchers'
        assert config_errors[1].message == 'datasets section is not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets'

    def test_empty_launchers_and_datasets_config(self):
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [], 'datasets': []}]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.launchers'
        assert config_errors[1].message == 'datasets section is not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets'

    def test_launcher_config_without_framework(self):
        launcher_config = {'model': 'foo'}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'framework is not provided'
        assert config_errors[0].entry == launcher_config
        assert config_errors[0].field_uri == 'models.launchers.0'
        assert config_errors[1].message == 'datasets section is not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets'

    def test_unregistered_launcher_config(self):
        launcher_config = {'framework': 'foo'}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launcher foo is not unregistered'
        assert config_errors[0].entry == launcher_config
        assert config_errors[0].field_uri == 'models.launchers.0'
        assert config_errors[1].message == 'datasets section is not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets'

    @pytest.mark.usefixtures('mock_file_exists')
    def test_valid_launcher_config(self):
        launcher_config = {'model': 'foo', 'framework': 'dlsdk', 'device': 'cpu'}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 1
        assert config_errors[0].message == 'datasets section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.datasets'

    @pytest.mark.usefixtures('mock_file_exists')
    def test_ignore_dataset_config(self):
        launcher_config = {'model': 'foo', 'framework': 'dlsdk', 'device': 'cpu'}
        config_errors = ModelEvaluator.validate_config(
            {'models': [{'launchers': [launcher_config], 'datasets': []}]}, delayed_annotation_loading=True
        )
        assert not config_errors

    @pytest.mark.usefixtures('mock_file_exists')
    def test_input_without_type(self):
        launcher_config = {'model': 'foo', 'framework': 'dlsdk', 'device': 'cpu', 'inputs': [{"name": 'input'}]}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 2
        assert config_errors[0].message.endswith('input type is not provided')
        assert config_errors[0].field_uri == 'models.launchers.0.inputs.0'
        assert config_errors[1].message == 'datasets section is not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets'

    @pytest.mark.usefixtures('mock_file_exists')
    def test_input_with_invalid_type(self):
        launcher_config = {'model': 'foo', 'framework': 'dlsdk', 'device': 'cpu', 'inputs': [{"name": 'input', 'type': 'FOO'}]}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 2
        assert config_errors[0].message.endswith('undefined input type FOO')
        assert config_errors[0].field_uri == 'models.launchers.0.inputs.0'
        assert config_errors[1].message == 'datasets section is not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets'

    @pytest.mark.usefixtures('mock_file_exists')
    def test_input_without_name(self):
        launcher_config = {'model': 'foo', 'framework': 'dlsdk', 'device': 'cpu', 'inputs': [{"type": 'INPUT'}]}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 2
        assert config_errors[0].message.endswith('input name is not provided')
        assert config_errors[0].field_uri == 'models.launchers.0.inputs.0'
        assert config_errors[1].message == 'datasets section is not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets'

    @pytest.mark.usefixtures('mock_file_exists')
    def test_adapter_str_config(self):
        launcher_config = {'model': 'foo', 'framework': 'dlsdk', 'device': 'cpu', 'adapter': 'classification'}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 1
        assert config_errors[0].message == 'datasets section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.datasets'

    @pytest.mark.usefixtures('mock_file_exists')
    def test_adapter_dict_config(self):
        launcher_config = {'model': 'foo', 'framework': 'dlsdk', 'device': 'cpu', 'adapter': {'type': 'classification'}}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 1
        assert config_errors[0].message == 'datasets section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.datasets'

    @pytest.mark.usefixtures('mock_file_exists')
    def test_unregistered_adapter_config(self):
        launcher_config = {'model': 'foo', 'framework': 'dlsdk', 'device': 'cpu', 'adapter': 'not_classification'}
        config_errors = ModelEvaluator.validate_config({'models': [{'launchers': [launcher_config], 'datasets': []}]})
        assert len(config_errors) == 2
        assert config_errors[0].message.startswith('Invalid value "not_classification"')
        assert config_errors[0].entry == 'not_classification'
        assert config_errors[0].field_uri.startswith('models.launchers.0') and config_errors[0].field_uri.endswith('adapter')
        assert config_errors[1].message == 'datasets section is not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_dataset_config_without_metrics(self):
        dataset_config = {'name': 'dataset', 'data_source': 'data', 'annotation': 'annotation'}
        config_errors = ModelEvaluator.validate_config({'models': [{'datasets': [dataset_config]}]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.launchers'
        assert config_errors[1].message == 'Metrics are not provided'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets.0'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_dataset_config_ignore_without_metrics(self):
        dataset_config = {'name': 'dataset', 'data_source': 'data', 'annotation': 'annotation'}
        config_errors = ModelEvaluator.validate_config(
            {'models': [{'datasets': [dataset_config]}]}, delayed_annotation_loading=True
        )
        assert len(config_errors) == 1
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.launchers'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_data_reader_without_data_source(self):
        dataset_config = {'name': 'dataset', 'annotation': 'annotation', 'metrics': [{'type': 'accuracy'}]}
        config_errors = ModelEvaluator.validate_config({'models': [{'datasets': [dataset_config]}]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.launchers'
        assert config_errors[1].message == 'Invalid value "None" for models.datasets.0.data_source: models.datasets.0.data_source is not allowed to be None'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets.0.data_source'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_data_reader_ignore_without_data_source(self):
        dataset_config = {'name': 'dataset', 'annotation': 'annotation', 'metrics': [{'type': 'accuracy'}]}
        config_errors = ModelEvaluator.validate_config({'models': [{'datasets': [dataset_config]}]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.launchers'
        assert config_errors[1].message == 'Invalid value "None" for models.datasets.0.data_source: models.datasets.0.data_source is not allowed to be None'
        assert not config_errors[1].entry
        assert config_errors[1].field_uri == 'models.datasets.0.data_source'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_unregistered_data_reader(self):
        dataset_config = {
            'name': 'dataset', 'annotation': 'annotation', 'metrics': [{'type': 'accuracy'}], 'reader': 'unknown'
        }
        config_errors = ModelEvaluator.validate_config({'models': [{'datasets': [dataset_config]}]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.launchers'
        assert config_errors[1].message.startswith('Invalid value "unknown" for models.datasets.0.reader')
        assert config_errors[-1].entry == 'unknown'
        assert config_errors[-1].field_uri == 'models.datasets.0.reader'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_unregistered_data_reader_with_postpone_data_loading(self):
        dataset_config = {
            'name': 'dataset', 'annotation': 'annotation', 'metrics': [{'type': 'accuracy'}], 'reader': 'unknown'
        }
        config_errors = ModelEvaluator.validate_config(
            {'models': [{'datasets': [dataset_config]}]}, delayed_annotation_loading=True
        )
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'models.launchers'
        assert config_errors[1].message.startswith('Invalid value "unknown" for models.datasets.0.reader')
        assert config_errors[-1].entry == 'unknown'
        assert config_errors[-1].field_uri == 'models.datasets.0.reader'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_str_data_reader(self):
        dataset_config = {
            'name': 'dataset', 'annotation': 'annotation', 'metrics': [{'type': 'accuracy'}],
            'reader': 'opencv_imread', 'data_source': 'data'
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 1
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_dict_data_reader(self):
        dataset_config = {
            'name': 'dataset', 'annotation': 'annotation', 'metrics': [{'type': 'accuracy'}],
            'reader': {'type': 'opencv_imread'}, 'data_source': 'data'
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 1
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'

    def test_data_source_does_not_exists(self):
        dataset_config = {'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data_dir'}
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 3
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[-1].message == 'Invalid value "data_dir" for datasets.0.data_source: path does not exist'
        assert config_errors[-1].entry == 'data_dir'
        assert config_errors[-1].field_uri == 'datasets.0.data_source'

    @pytest.mark.usefixtures('mock_file_exists')
    def test_data_source_is_file(self):
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'annotation': 'annotation', 'data_source': 'data'
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == 'Invalid value "data" for datasets.0.data_source: path is not a directory'
        assert config_errors[1].entry == 'data'
        assert config_errors[1].field_uri == 'datasets.0.data_source'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_annotation_is_not_provided(self):
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data'
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == 'annotation_conversion or annotation field should be provided'
        assert config_errors[1].entry == dataset_config
        assert config_errors[1].field_uri == 'datasets.0'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_annotation_conversion_without_converter(self):
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation_conversion': {}
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == 'converter is not found'
        assert config_errors[1].entry == {}
        assert config_errors[1].field_uri == 'datasets.0.annotation_conversion'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_annotation_conversion_missed_parameter(self):
        conversion_parameters = {'converter': 'imagenet'}
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation_conversion': conversion_parameters
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == 'Invalid config for datasets.0.annotation_conversion: missing required fields: annotation_file'
        assert config_errors[1].entry == conversion_parameters
        assert config_errors[1].field_uri == 'datasets.0.annotation_conversion'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_annotation_conversion_extra_parameter(self):
        conversion_parameters = {'converter': 'imagenet', 'annotation_file': 'file', 'something_extra': 'extra'}
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation_conversion': conversion_parameters
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == "datasets.0.annotation_conversion specifies unknown options: ['something_extra']"
        assert config_errors[1].entry == conversion_parameters
        assert config_errors[1].field_uri == 'datasets.0.annotation_conversion'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_annotation_conversion_config(self):
        conversion_parameters = {'converter': 'imagenet', 'annotation_file': 'file'}
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation_conversion': conversion_parameters
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 1
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_preprocessing_config(self):
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'preprocessing': [{'type': 'auto_resize'}]
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 1
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_preprocessing_config_unknown_type(self):
        preprocessing_config = [{'type': 'bgr_to_rgb'}, {'type': 'unknown', 'size': 224}]
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'preprocessing': preprocessing_config
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == 'preprocessor unknown unregistered'
        assert config_errors[1].entry == preprocessing_config[1]
        assert config_errors[1].field_uri == 'datasets.0.preprocessing.1'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_preprocessing_config_unknown_type_with_postponed_annotation(self):
        preprocessing_config = [{'type': 'bgr_to_rgb'}, {'type': 'unknown', 'size': 224}]
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'preprocessing': preprocessing_config
        }
        config_errors = ModelEvaluator.validate_config(
            {'datasets': [dataset_config]}, delayed_annotation_loading=True
        )
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == 'preprocessor unknown unregistered'
        assert config_errors[1].entry == preprocessing_config[1]
        assert config_errors[1].field_uri == 'datasets.0.preprocessing.1'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_preprocessing_config_extra_parameter(self):
        preprocessing_config = [{'type': 'bgr_to_rgb'}, {'type': 'resize', 'size': 224, 'something_extra': True}]
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'preprocessing': preprocessing_config
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == "datasets.0.preprocessing.1 specifies unknown options: ['something_extra']"
        assert config_errors[1].entry == preprocessing_config[1]
        assert config_errors[1].field_uri == 'datasets.0.preprocessing.1'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_preprocessing_config_unknown_parameter(self):
        preprocessing_config = [{'type': 'bgr_to_rgb'}, {'type': 'not_resize', 'size': 224}]
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'preprocessing': preprocessing_config
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == "preprocessor not_resize unregistered"
        assert config_errors[1].entry == preprocessing_config[1]
        assert config_errors[1].field_uri == 'datasets.0.preprocessing.1'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_postprocessing_config(self):
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'postprocessing': [{'type': 'resize_prediction_boxes'}]
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 1
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_postprocessing_config_unknown_type(self):
        postprocessing_config = [{'type': 'unknown', 'size': 224}]
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'postprocessing': postprocessing_config
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == 'postprocessor unknown unregistered'
        assert config_errors[1].entry == postprocessing_config[0]
        assert config_errors[1].field_uri == 'datasets.0.postprocessing.0'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_postprocessing_config_extra_parameter(self):
        postprocessing_config = [{'type': 'resize_prediction_boxes', 'something_extra': True}]
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'postprocessing': postprocessing_config
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == "datasets.0.postprocessing.0 specifies unknown options: ['something_extra']"
        assert config_errors[1].entry == postprocessing_config[0]
        assert config_errors[1].field_uri == 'datasets.0.postprocessing.0'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_postprocessing_config_extra_parameter_with_postponed_annotation(self):
        postprocessing_config = [{'type': 'resize_prediction_boxes', 'something_extra': True}]
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'postprocessing': postprocessing_config
        }
        config_errors = ModelEvaluator.validate_config(
            {'datasets': [dataset_config]}, delayed_annotation_loading=True
        )
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == "datasets.0.postprocessing.0 specifies unknown options: ['something_extra']"
        assert config_errors[1].entry == postprocessing_config[0]
        assert config_errors[1].field_uri == 'datasets.0.postprocessing.0'

    @pytest.mark.usefixtures('mock_path_exists')
    def test_postprocessing_config_unknown_parameter(self):
        postprocessing_config = [{'type': 'bgr_to_rgb'}]
        dataset_config = {
            'name': 'dataset', 'metrics': [{'type': 'accuracy'}], 'data_source': 'data',
            'annotation': 'annotation', 'postprocessing': postprocessing_config
        }
        config_errors = ModelEvaluator.validate_config({'datasets': [dataset_config]})
        assert len(config_errors) == 2
        assert config_errors[0].message == 'launchers section is not provided'
        assert not config_errors[0].entry
        assert config_errors[0].field_uri == 'launchers'
        assert config_errors[1].message == "postprocessor bgr_to_rgb unregistered"
        assert config_errors[1].entry == postprocessing_config[0]
        assert config_errors[1].field_uri == 'datasets.0.postprocessing.0'


class TestValidationScheme:
    def test_common_validation_scheme(self):
        validation_scheme = ModelEvaluator.validation_scheme()
        assert isinstance(validation_scheme, dict)
        assert len(validation_scheme) == 1
        assert 'models' in validation_scheme
        assert len(validation_scheme['models']) == 1
        assert contains_all(validation_scheme['models'][0], ['name', 'launchers', 'datasets'])
        assert isinstance(validation_scheme['models'][0]['name'], StringField)
        model_validation_scheme = validation_scheme['models'][0]
        assert model_validation_scheme['launchers'].__name__ == Launcher.__name__
        assert model_validation_scheme['datasets'].__name__ == Dataset.__name__
        assert isinstance(model_validation_scheme['launchers'].validation_scheme(), list)
        assert isinstance(model_validation_scheme['datasets'].validation_scheme(), list)

    def test_dataset_validation_scheme(self):
        dataset_validation_scheme = Dataset.validation_scheme()
        assert isinstance(dataset_validation_scheme, list)
        assert len(dataset_validation_scheme) == 1
        dataset_params = [key for key in Dataset.parameters() if not key.startswith('_')]
        assert isinstance(dataset_validation_scheme[0], dict)
        assert contains_all(dataset_validation_scheme[0], dataset_params)
        assert len(dataset_validation_scheme[0]) == len(dataset_params)
        assert isinstance(dataset_validation_scheme[0]['name'], StringField)
        assert isinstance(dataset_validation_scheme[0]['annotation'], PathField)
        assert isinstance(dataset_validation_scheme[0]['data_source'], PathField)
        assert isinstance(dataset_validation_scheme[0]['dataset_meta'], PathField)
        assert isinstance(dataset_validation_scheme[0]['subsample_size'], BaseField)
        assert isinstance(dataset_validation_scheme[0]['shuffle'], BoolField)
        assert isinstance(dataset_validation_scheme[0]['subsample_seed'], NumberField)
        assert isinstance(dataset_validation_scheme[0]['analyze_dataset'], BoolField)
        assert isinstance(dataset_validation_scheme[0]['segmentation_masks_source'], PathField)
        assert isinstance(dataset_validation_scheme[0]['additional_data_source'], PathField)
        assert isinstance(dataset_validation_scheme[0]['batch'], NumberField)

        assert dataset_validation_scheme[0]['reader'] == BaseReader
        assert dataset_validation_scheme[0]['preprocessing'] == Preprocessor
        assert dataset_validation_scheme[0]['postprocessing'] == Postprocessor
        assert dataset_validation_scheme[0]['metrics'] == Metric
        assert dataset_validation_scheme[0]['annotation_conversion'] == BaseFormatConverter

    def test_launcher_validation_scheme(self):
        launchers_validation_scheme = Launcher.validation_scheme()
        assert isinstance(launchers_validation_scheme, list)
        assert len(launchers_validation_scheme) == len(Launcher.providers)
        dlsdk_launcher_val_scheme = Launcher.validation_scheme('dlsdk')
        assert dlsdk_launcher_val_scheme['adapter'] == Adapter

    def test_adapter_validation_scheme(self):
        adapter_full_validation_scheme = Adapter.validation_scheme()
        assert isinstance(adapter_full_validation_scheme, dict)
        assert len(adapter_full_validation_scheme) ==  len(Adapter.providers)
        assert contains_all(adapter_full_validation_scheme, Adapter.providers)
        assert set(adapter_full_validation_scheme['classification']) == set(Adapter.validation_scheme('classification'))

    def test_metric_validation_scheme(self):
        metrics_full_validation_scheme = Metric.validation_scheme()
        assert isinstance(metrics_full_validation_scheme, list)
        assert len(metrics_full_validation_scheme) == len(Metric.providers)
        accuracy_validation_scheme = Metric.validation_scheme('accuracy')
        assert isinstance(accuracy_validation_scheme, dict)
        assert contains_all(accuracy_validation_scheme, ['type', 'top_k'])
        assert isinstance(accuracy_validation_scheme['type'], StringField)
        assert isinstance(accuracy_validation_scheme['top_k'], NumberField)

    def test_postprocessing_validation_scheme(self):
        postprocessing_validation_scheme = Postprocessor.validation_scheme()
        assert isinstance(postprocessing_validation_scheme, list)
        assert len(postprocessing_validation_scheme) == len(Postprocessor.providers)
        resize_pred_boxes_scheme = Postprocessor.validation_scheme('resize_prediction_boxes')
        assert isinstance(resize_pred_boxes_scheme, dict)
        assert contains_all(resize_pred_boxes_scheme, ['type', 'rescale'])
        assert isinstance(resize_pred_boxes_scheme['type'], StringField)
        assert isinstance(resize_pred_boxes_scheme['rescale'], BoolField)

    def test_preprocessing_validation_scheme(self):
        preprocessing_validation_scheme = Preprocessor.validation_scheme()
        assert isinstance(preprocessing_validation_scheme, list)
        assert len(preprocessing_validation_scheme) == len(Preprocessor.providers)
        auto_resize_scheme = Preprocessor.validation_scheme('auto_resize')
        assert isinstance(auto_resize_scheme, dict)
        assert contains_all(auto_resize_scheme, ['type'])
        assert isinstance(auto_resize_scheme['type'], StringField)

    def test_annotation_conversion_validation_scheme(self):
        converter_validation_scheme = BaseFormatConverter.validation_scheme()
        assert isinstance(converter_validation_scheme, dict)
        assert len(converter_validation_scheme) == len(BaseFormatConverter.providers)
        assert contains_all(converter_validation_scheme, BaseFormatConverter.providers)
        assert set(converter_validation_scheme['imagenet']) == set(BaseFormatConverter.validation_scheme('imagenet'))
