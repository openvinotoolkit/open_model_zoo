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

import enum
import numpy as np

from ..utils import UnsupportedPackage

from ..representation import TimeSeriesForecastingAnnotation
from ..config import PathField, NumberField
from .format_converter import BaseFormatConverter, ConverterReturn
from ..logging import print_info
try:
    import pandas as pd
except ImportError as import_error:
    pd = UnsupportedPackage("pandas", import_error.msg)

try:
    import sklearn.preprocessing as sk_preprocessing
except ImportError as error:
    sk_preprocessing = UnsupportedPackage('sklearn', error.msg)

try:
    from tqdm import tqdm
except ImportError as error:
    tqdm = UnsupportedPackage('tqdm', error.msg)


def expand(x, axis=0):
    return np.expand_dims(x, axis=axis)


def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.
    Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
    """

    cols = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(cols) != 1:
        raise ValueError('Invalid number of columns for {}'.format(input_type))

    return cols[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.
    Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude
    Returns:
    List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


def aggregating_to_hourly_data(df):
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Used to determine the start and end dates of a series
    output = df.resample('1h').mean().replace(0., np.nan)

    earliest_time = output.index.min()

    df_list = []
    for label in output:
        print_info('Processing {}'.format(label))
        srs = output[label]

        start_date = min(srs.fillna(method='ffill').dropna().index)
        end_date = max(srs.fillna(method='bfill').dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.)

        tmp = pd.DataFrame({'power_usage': srs})
        date = tmp.index
        tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
        tmp['days_from_start'] = (date - earliest_time).days
        tmp['categorical_id'] = label
        tmp['date'] = date
        tmp['id'] = label
        tmp['hour'] = date.hour
        tmp['day'] = date.day
        tmp['day_of_week'] = date.dayofweek
        tmp['month'] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['t']
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096) & (output['days_from_start'] < 1346)].copy()
    return output


def get_index_filtering(data, id_col, target_col, lookback):
    g = data.groupby(id_col)

    df_index_abs = g[target_col].transform(lambda x: x.index+lookback).reset_index().rename(
        columns={'index': 'init_abs', target_col[0]: 'end_abs'})
    df_index_rel_init = g[target_col].transform(lambda x: x.reset_index(drop=True).index).rename(
        columns={target_col[0]: 'init_rel'})
    df_index_rel_end = g[target_col].transform(lambda x: x.reset_index(drop=True).index+lookback).rename(
        columns={target_col[0]: 'end_rel'})
    df_total_count = g[target_col].transform(lambda x: x.shape[0] - lookback + 1).rename(
        columns={target_col[0]: 'group_count'})

    return pd.concat([df_index_abs,
                      df_index_rel_init,
                      df_index_rel_end,
                      data[id_col],
                      df_total_count], axis=1).reset_index(drop=True)

def get_time_steps():
    return get_fixed_params()['total_time_steps']

def get_num_encoder_steps():
    return get_fixed_params()['num_encoder_steps']


# Default params
def get_fixed_params():
    """Returns fixed model parameters for experiments."""
    fixed_params = {
        'total_time_steps': 8 * 24,
        'num_encoder_steps': 7 * 24,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }
    return fixed_params

# Type defintions
class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""
    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index


class ElectricityFormatter:
    """Defines and formats data for the electricity dataset.
    Note that per-entity z-score normalization is used here, and is implemented
    across functions.
    Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
    """

    column_definition = [
        ('id', DataTypes.REAL_VALUED, InputTypes.ID),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self._time_steps = get_fixed_params()['total_time_steps']
        self._num_encoder_steps = get_fixed_params()['num_encoder_steps']

    def split_data(self, df, valid_boundary=1315, test_boundary=1339):
        """Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split.
        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data
        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print_info('Formatting train-valid-test splits.')

        index = df['days_from_start']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7) & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.
            Args:
              df: Data to use to calibrate scalers.
        """
        print_info('Setting scalers with training data...')

        column_definitions = self.getcolumn_definition()
        id_column = get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        # Format real scalers
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Initialise scaler caches
        self.real_scalers = {}
        self.target_scaler = {}
        identifiers = []
        for identifier, sliced in df.groupby(id_column):

            if len(sliced) >= self._time_steps:

                data = sliced[real_inputs].values
                targets = sliced[[target_column]].values
                self.real_scalers[identifier] = sk_preprocessing.StandardScaler().fit(data)
                self.target_scaler[identifier] = sk_preprocessing.StandardScaler().fit(targets)
                identifiers.append(identifier)

        # Format categorical scalers
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sk_preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

        # Extract identifiers in case required
        self.identifiers = identifiers

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data frame.
        """

        if self.real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Extract relevant columns
        column_definitions = self.getcolumn_definition()
        id_col = get_single_col_by_input_type(InputTypes.ID, column_definitions)
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Transform real inputs per entity
        df_list = []
        for identifier, sliced in df.groupby(id_col):
            # Filter out any trajectories that are too short
            if len(sliced) >= self._time_steps:
                sliced_copy = sliced.copy()
                sliced_copy[real_inputs] = self.real_scalers[identifier].transform(sliced_copy[real_inputs].values)
                df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def getcolumn_definition(self):
        """"Returns formatted column definition in order expected by the TFT."""

        column_definition = self.column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        def _check_single_column(input_type):

            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError('Illegal number of inputs ({}) of type {}'.format(
                    length, input_type))

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED and
            tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]
        categorical_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.CATEGORICAL and
            tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        return identifier + time + real_inputs + categorical_inputs


class ElectricityTimeSeriesForecastingConverter(BaseFormatConverter):
    """ Annotation converter for electricity dataset.
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'
    """

    __provider__ = "electricity"
    annotation_types = (TimeSeriesForecastingAnnotation, )

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update({
            "data_path_file": PathField(description="Path to dataset file in .csv format."),
            "num_encoder_steps": NumberField(
                description='The maximum number of historical timestamps that model use.',
                optional=True, default=7 * 24, value_type=int
            )
        })

        return configuration_parameters

    def configure(self):
        if isinstance(pd, UnsupportedPackage):
            pd.raise_error(self.__provider__)
        if isinstance(sk_preprocessing, UnsupportedPackage):
            sk_preprocessing.raise_error(self.__provider__)
        self.data_path_file = self.get_value_from_config('data_path_file')
        self.num_encoder_steps = int(self.get_value_from_config('num_encoder_steps'))
        self.formatter = ElectricityFormatter()

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        _, _, data = self.formatter.split_data(
            aggregating_to_hourly_data(pd.read_csv(self.data_path_file, index_col=0, sep=';', decimal=','))
        )
        data = data.reset_index(drop=True)
        data_index, col_mappings = self.build_data_index(data)
        samples = []
        iterator = range(int(data_index.shape[0]))
        if not isinstance(tqdm, UnsupportedPackage):
            iterator = tqdm(iterator)
        for idx in iterator:
            samples.append(
                TimeSeriesForecastingAnnotation(f"inputs_{idx}", *self.get_sample(data, data_index, col_mappings, idx))
            )

        return ConverterReturn(samples, None, None)

    def build_data_index(self, data):
        column_definition = self.formatter.column_definition
        col_mappings = {
            'identifier': [get_single_col_by_input_type(InputTypes.ID, column_definition)],
            'time': [get_single_col_by_input_type(InputTypes.TIME, column_definition)],
            'outputs': [get_single_col_by_input_type(InputTypes.TARGET, column_definition)],
            'inputs': [tup[0] for tup in column_definition if tup[2] not in {InputTypes.ID, InputTypes.TIME}]
        }
        lookback = get_time_steps()
        data_index = get_index_filtering(data, col_mappings["identifier"], col_mappings["outputs"], lookback)
        group_size = data.groupby(col_mappings["identifier"]).apply(lambda x: x.shape[0]).mean()
        data_index = data_index[data_index.end_rel < group_size].reset_index()
        return data_index, col_mappings

    def get_sample(self, data, data_index, col_mappings, idx):
        _data_index = data.iloc[data_index.init_abs.iloc[idx]:data_index.end_abs.iloc[idx]]

        data_map = {}
        for k in col_mappings:
            cols = col_mappings[k]

            if k not in data_map:
                data_map[k] = [_data_index[cols].values]
            else:
                data_map[k].append(_data_index[cols].values)

        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

        # prepare outputs
        scaler = self.formatter.target_scaler[data_map["identifier"][0][0]]
        outputs = data_map['outputs'][self.num_encoder_steps:, 0]
        return expand(data_map['inputs']), expand(outputs), scaler.mean_, scaler.scale_
