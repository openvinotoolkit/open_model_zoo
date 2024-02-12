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

import struct
import wave

import numpy as np

from ..config import BoolField, StringField
from .data_reader import BaseReader, DataRepresentation, KaldiMatrixIdentifier, KaldiFrameIdentifier
from ..utils import contains_any, UnsupportedPackage

try:
    import soundfile as sf
except (ImportError, OSError) as import_error:
    sf = UnsupportedPackage('soundfile', str(import_error))


class WavReader(BaseReader):
    __provider__ = 'wav_reader'

    _samplewidth_types = {
        1: np.uint8,
        2: np.int16
    }

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'mono': BoolField(optional=True, default=False,
                              description='get mean along channels if multichannel audio loaded'),
            'to_float': BoolField(optional=True, default=False, description='converts audio signal to float'),
            'float_dtype': StringField(
                choices=['float16', 'float32', 'float64'], optional=True, default='float32',
                description='specifies precision for conversion to float '
            ),
            'flattenize': BoolField(optional=True, default=False, description='flattenize signal')
        })
        return params

    def configure(self):
        super().configure()
        self.mono = self.get_value_from_config('mono')
        self.to_float = self.get_value_from_config('to_float')
        self.float_dtype = self.get_value_from_config('float_dtype')
        if self.float_dtype == 'float64':
            self.float_dtype = 'float'
        self.flattenize = self.get_value_from_config('flattenize')

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        with wave.open(str(data_path), "rb") as wav:
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            nframes = wav.getnframes()
            data = wav.readframes(nframes)
            if self._samplewidth_types.get(sample_width):
                data = np.frombuffer(data, dtype=self._samplewidth_types[sample_width])
            else:
                raise RuntimeError("Reader {} couldn't process file {}: unsupported sample width {}"
                                   "(reader only supports {})"
                                   .format(self.__provider__, self.data_source / data_id,
                                           sample_width, [*self._samplewidth_types.keys()]))
            channels = wav.getnchannels()

            data = data.reshape(-1, channels).T
            if channels > 1 and self.mono:
                data = data.mean(0, keepdims=True)
            if self.to_float:
                data = data.astype(self.float_dtype) / np.iinfo(self._samplewidth_types[sample_width]).max

            if self.flattenize:
                data = data.flatten()

        return data, {'sample_rate': sample_rate}

    def read_item(self, data_id):
        return DataRepresentation(*self.read_dispatcher(data_id), identifier=data_id)


class KaldiARKReader(BaseReader):
    __provider__ = 'kaldi_ark_reader'

    def configure(self):
        super().configure()
        self.buffer = {}

    @staticmethod
    def read_frames(in_file):
        ut = {}
        with open(str(in_file), 'rb') as fd:
            while True:
                try:
                    key = KaldiARKReader.read_token(fd)
                    if not key:
                        break
                    binary = fd.read(2).decode()
                    if binary in [' [', '[\r']:
                        mat = KaldiARKReader.read_ascii_mat(fd)
                    else:
                        ark_type = KaldiARKReader.read_token(fd)
                        float_size = 4 if ark_type[0] == 'F' else 8
                        float_type = np.float32 if ark_type[0] == 'F' else float
                        num_rows = KaldiARKReader.read_int32(fd)
                        num_cols = KaldiARKReader.read_int32(fd)
                        mat_data = fd.read(float_size * num_cols * num_rows)
                        mat = np.frombuffer(mat_data, dtype=float_type).reshape(num_rows, num_cols)
                    ut[key] = mat
                except EOFError:
                    break
            return ut

    def read_utterance(self, file_name, utterance):
        if file_name not in self.buffer:
            self.buffer[file_name] = self.read_frames(self.data_source / file_name)
        return self.buffer[file_name][utterance]

    def read_frame(self, file_name, utterance, idx):
        return self.read_utterance(file_name, utterance)[idx]

    @staticmethod
    def read_int32(fd):
        int_size = bytes.decode(fd.read(1))
        assert int_size == '\04', 'Expect \'\\04\', but gets {}'.format(int_size)
        int_str = fd.read(4)
        int_val = struct.unpack('i', int_str)
        return int_val[0]

    @staticmethod
    def read_token(fd):
        key = ''
        while True:
            c = bytes.decode(fd.read(1))
            if c in [' ', '', '\0', '\4']:
                break
            key += c
        return None if key == '' else key.strip()

    def read(self, data_id, reset=True):
        assert (
            isinstance(data_id, (KaldiMatrixIdentifier, KaldiFrameIdentifier))
        ), "Kaldi reader support only Kaldi specific data IDs"
        file_id = data_id.file
        if file_id not in self.buffer and self.buffer and reset:
            self.reset()
        if len(data_id) == 3:
            return self.read_frame(data_id.file, data_id.key, data_id.id)
        matrix = self.read_utterance(data_id.file, data_id.key)
        if self.multi_infer:
            matrix = list(matrix)
        return matrix

    def _read_list(self, data_id):
        if not contains_any(self.buffer, [data.file for data in data_id]) and self.buffer:
            self.reset()

        return [self.read(idx, reset=False) for idx in data_id]

    def reset(self):
        del self.buffer
        self.buffer = {}

    @staticmethod
    def read_ascii_mat(fd):
        rows = []
        while True:
            line = fd.readline().decode()
            if not line.strip():
                continue # skip empty line
            arr = line.strip().split()
            if arr[-1] != ']':
                rows.append(np.array(arr, dtype='float32')) # not last line
            else:
                rows.append(np.array(arr[:-1], dtype='float32')) # last line
                mat = np.vstack(rows)
                return mat


class FlacReader(BaseReader):
    __provider__ = 'flac_reader'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'dtype': StringField(
                choices=['float32', 'float64', 'int16', 'int32'], optional=True, default='int16',
                description='specifies precision of reading data'
            ),
            'mono': BoolField(optional=True, default=False,
                              description='get mean along channels if multichannel audio loaded'),
        })
        return params

    def configure(self):
        if isinstance(sf, UnsupportedPackage):
            sf.raise_error(self.__provider__)
        super().configure()
        self.dtype = self.get_value_from_config('dtype')
        self.mono = self.get_value_from_config('mono')

    @staticmethod
    def prepare_read(frames, start=0, stop=None):
        start, stop, _ = slice(start, stop).indices(frames)
        stop = max(stop, start)
        res_frames = stop - start
        return res_frames

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id

        with sf.SoundFile(data_path) as f:
            frames = self.prepare_read(f.frames)
            data = f.read(frames, dtype=self.dtype)
            samplerate = f.samplerate
            channels = f.channels

        data = data.reshape(-1, channels).T
        if channels > 1 and self.mono:
            data = data.mean(0, keepdims=True)
        return data, {'sample_rate': samplerate}

    def read_item(self, data_id):
        return DataRepresentation(*self.read_dispatcher(data_id), identifier=data_id)
