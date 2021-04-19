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

import os
import struct
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from .adapter import Adapter
from ..config import PathField, BoolField, NumberField, ConfigError
from ..representation import CharacterRecognitionPrediction
from ..utils import read_txt


class KaldiLatGenDecoder(Adapter):
    __provider__ = 'kaldi_latgen_faster_mapped'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'fst_file': PathField(description='WFST state graph file'),
            'words_file': PathField(description='words table file'),
            'transition_model_file': PathField(description='transition model file'),
            'beam': NumberField(optional=True, value_type=int, min_value=1, description='beam size'),
            'lattice_beam': NumberField(optional=True, value_type=int, min_value=1, description='lattice beam size'),
            'allow_partial': BoolField(optional=True, default=True, description='allow decoding'),
            'acoustic_scale': NumberField(optional=True, default=0.1, value_type=float),
            'min_active': NumberField(optional=True, value_type=int, min_value=9),
            'max_active': NumberField(optional=True, value_type=int, min_value=0),
            'inverse_acoustic_scale': NumberField(optional=True, value_type=float, default=0),
            'word_insertion_penalty': NumberField(optional=True, default=0, value_type=float),
            'kaldi_bin_dir': PathField(is_directory=True)
        })
        return params

    def configure(self):
        self.fst_file = self.get_value_from_config('fst_file')
        self.words_file = self.get_value_from_config('words_file')
        self.transition_model = self.get_value_from_config('transition_model_file')
        self.words_table = self.read_words_table()
        self.beam = self.get_value_from_config('beam')
        self.lattice_beam = self.get_value_from_config('lattice_beam')
        self.min_active = self.get_value_from_config('min_active')
        self.max_active = self.get_value_from_config('max_active')
        self.acoustic_scale = self.get_value_from_config('acoustic_scale')
        self.inv_acoustic_scale = self.get_value_from_config('inverse_acoustic_scale')
        self.word_insertion_penalty = self.get_value_from_config('word_insertion_penalty')
        self.allow_partial = self.get_value_from_config('allow_partial')
        self.create_cmd()

    def read_words_table(self):
        words_table = {}
        for line in read_txt(self.words_file):
            word, idx = line.split()
            words_table[int(idx)] = word
        return words_table

    def create_cmd(self):
        self.kaldi_bin_dir = self.get_value_from_config('kaldi_bin_dir')
        error_msg = 'Path to Kaldi executable {} is not found'
        executable = '{}' if os.name != 'nt' else '{}.exe'
        latgen_path = self.kaldi_bin_dir / executable.format('latgen-faster-mapped')
        if not latgen_path.exists():
            raise ConfigError(error_msg.format(latgen_path))
        latgen_cmd = ' '.join([str(latgen_path),
                      "--min-active={}".format(self.min_active),
                      "--max-active={}".format(self.max_active),
                      "--max-mem=50000000",
                      "--beam={}".format(str(self.beam)),
                      "--lattice-beam={}".format(str(self.lattice_beam)),
                      "--acoustic-scale={}".format(str(self.acoustic_scale)),
                      "--allow-partial={}".format(self.allow_partial),
                      "--word-symbol-table={}".format(self.words_file),
                      str(self.transition_model), str(self.fst_file),
                      "ark:{}", "ark:-"])

        lattice_scale_path = self.kaldi_bin_dir / executable.format('lattice-scale')
        if not lattice_scale_path.exists():
            raise ConfigError(error_msg.format(lattice_scale_path))
        scale_cmd = '{} --inv-acoustic-scale={} ark:- ark:-'.format(lattice_scale_path, self.inv_acoustic_scale)
        lattice_best_path = self.kaldi_bin_dir / executable.format('lattice-best-path')
        if not lattice_best_path.exists():
            raise ConfigError(error_msg.format(lattice_best_path))
        best_path_cmd = '{} --word-symbol-table={} ark:- ark,t:-'.format(lattice_best_path, self.words_file)

        lattice_add_penalty_path = self.kaldi_bin_dir / executable.format('lattice-add-penalty')
        if not lattice_add_penalty_path.exists():
            raise ConfigError(error_msg.format(lattice_add_penalty_path))
        add_penalty_cmd = '{} --word-ins-penalty={} ark:- ark:-'.format(
            lattice_add_penalty_path, self.word_insertion_penalty
        )
        self.decoder_cmd = ' | '.join([latgen_cmd, scale_cmd, add_penalty_cmd, best_path_cmd])
        self._temp_dir = tempfile.TemporaryDirectory(suffix=self.__provider__, dir=Path.cwd())

    def reset(self):
        self._temp_dir.cleanup()

    def release(self):
        if Path(self._temp_dir.name).exists():
            self._temp_dir.cleanup()

    def process(self, raw, identifiers, frame_meta):
        results = []
        preds = self._extract_predictions(raw, frame_meta)
        for identifier, log_scores in zip(identifiers, preds[self.output_blob]):
            utt_name = identifier.key if not isinstance(identifier, list) else identifier[0].key
            scores_file = self.dump_scores(utt_name, log_scores)
            trans = self.run_decoder(scores_file)
            results.append(CharacterRecognitionPrediction(identifier, trans[utt_name]))
        return results

    def _extract_predictions(self, outputs_list, meta):
        is_multi_infer = meta[-1].get('multi_infer', False) if meta else False
        if not is_multi_infer:
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        output_map = {
            self.output_blob: np.expand_dims(np.concatenate([out[self.output_blob] for out in outputs_list], axis=0), 0)
        }

        return output_map

    def dump_scores(self, utterance_key, mat):
        out_file = Path(self._temp_dir.name) / '{}_scores.ark'.format(utterance_key)
        with out_file.open('wb') as fd:
            fd.write(str.encode(utterance_key + " "))
            fd.write(str.encode('\0B'))
            if mat.dtype not in [np.float32, np.float64]:
                raise RuntimeError("Unsupported numpy dtype: {}".format(mat.dtype))
            mat_type = 'FM' if mat.dtype == np.float32 else 'DM'
            fd.write(str.encode(mat_type + " "))
            num_rows, num_cols = mat.shape
            fd.write(str.encode('\04'))
            int_pack = struct.pack('i', num_rows)
            fd.write(int_pack)
            fd.write(str.encode('\04'))
            int_pack = struct.pack('i', num_cols)
            fd.write(int_pack)
            fd.write(mat.tobytes())
        return out_file

    def run_decoder(self, scores_file):

        def get_cmd_result(process):
            _, stderr = process.communicate()
            stderr = stderr.decode('utf-8') if stderr else None
            if process.returncode != 0:
                raise RuntimeError("\nAn error occurred!\n Return code: {}\n Error output:\n{}"
                                   .format(str(process.returncode), stderr))

        outfile = scores_file.with_suffix('.txt')
        with outfile.open('w') as f:
            p = subprocess.Popen(self.decoder_cmd.format(scores_file), stdout=f, stderr=subprocess.PIPE, shell=True)
            get_cmd_result(p)
        return self.get_transcript(outfile)

    def get_transcript(self, lattice_file):
        transcripts = {}
        for line in read_txt(lattice_file):
            result = line.split(' ')
            utt = result[0]
            decoded = ' '.join([self.words_table[int(idx)] for idx in result[1:]])
            transcripts[utt] = decoded
        return transcripts
