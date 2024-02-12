"""
 Copyright (c) 2020-2024 Intel Corporation

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

import logging as log

import numpy as np

from utils.text_preprocessing import text_to_sequence, _symbol_to_id
from utils.embeddings_processing import PCA


def check_input_name(model, input_tensor_name):
    try:
        model.input(input_tensor_name)
        return True
    except RuntimeError:
        return False


class ForwardTacotronIE:
    def __init__(self, model_duration, model_forward, core, device='CPU', verbose=False):
        self.verbose = verbose
        self.device = device
        self.core = core

        self.duration_predictor_model = self.load_network(model_duration)
        self.duration_predictor_request = self.create_infer_request(self.duration_predictor_model, model_duration)

        self.forward_model = self.load_network(model_forward)
        self.forward_request = self.create_infer_request(self.forward_model, model_forward)

        # fixed length of the sequence of symbols
        self.duration_len = self.duration_predictor_model.input('input_seq').shape[1]
        # fixed length of the input embeddings for forward
        self.forward_len = self.forward_model.input('data').shape[1]
        if self.verbose:
            log.debug('Forward limitations : {0} symbols and {1} embeddings'.format(self.duration_len, self.forward_len))
        self.is_attention = check_input_name(self.forward_model, "pos_mask")
        if self.is_attention:
            self.init_pos_mask()
        else:
            self.pos_mask = None

        self.is_multi_speaker = check_input_name(self.duration_predictor_model, "speaker_embedding")
        if self.is_multi_speaker:
            self.init_speaker_information()
        else:
            self.male_idx = None
            self.female_idx = None
            self.speaker_embeddings = None
            self.female_embeddings = None
            self.male_embeddings = None

    def init_pos_mask(self, mask_sz=6000, window_size=4):
        mask_arr = np.zeros((1, 1, mask_sz, mask_sz), dtype=np.float32)
        width = 2 * window_size + 1
        for i in range(mask_sz - width):
            mask_arr[0][0][i][i:i + width] = 1.0

        self.pos_mask = mask_arr

    @staticmethod
    def sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = np.max(length)
        x = np.arange(max_length, dtype=length.dtype)
        x = np.expand_dims(x, axis=(0))
        length = np.expand_dims(length, axis=(1))
        return x < length

    def seq_to_indexes(self, text):
        res = text_to_sequence(text)
        if self.verbose:
            log.debug(res)
        return res

    @staticmethod
    def build_index(duration, x):
        duration[np.where(duration < 0)] = 0
        tot_duration = np.cumsum(duration, 1)
        max_duration = int(tot_duration.max().item())
        index = np.zeros([x.shape[0], max_duration, x.shape[2]], dtype='long')

        for i in range(tot_duration.shape[0]):
            pos = 0
            for j in range(tot_duration.shape[1]):
                pos1 = tot_duration[i, j]
                index[i, pos:pos1, :] = j
                pos = pos1
            index[i, pos:, :] = j
        return index

    @staticmethod
    def gather(a, dim, index):
        expanded_index = [index if dim == i else np.arange(a.shape[i]).reshape(
                                                  [-1 if i == j else 1 for j in range(a.ndim)]) for i in range(a.ndim)]
        return a[tuple(expanded_index)]

    def load_network(self, model_path):
        log.info('Reading ForwardTacotron model {}'.format(model_path))
        return self.core.read_model(model_path)

    def create_infer_request(self, model, path):
        compiled_model = self.core.compile_model(model, device_name=self.device)
        log.info('The ForwardTacotron model {} is loaded to {}'.format(path, self.device))
        return compiled_model.create_infer_request()

    def infer_duration(self, sequence, speaker_embedding=None, alpha=1.0, non_empty_symbols=None):
        if self.is_attention:
            input_mask = self.sequence_mask(np.array([[non_empty_symbols]]), sequence.shape[1])
            pos_mask = self.pos_mask[:, :, :sequence.shape[1], :sequence.shape[1]]
            inputs = {"input_seq": sequence,
                      "input_mask": input_mask,
                      "pos_mask": pos_mask}
            if speaker_embedding is not None:
                inputs["speaker_embedding"] = np.array(speaker_embedding)
            self.duration_predictor_request.infer(inputs)
        else:
            self.duration_predictor_request.infer(inputs={"input_seq": sequence})
        duration = self.duration_predictor_request.get_tensor("duration").data[:] * alpha

        duration = (duration + 0.5).astype('int').flatten()
        duration = np.expand_dims(duration, axis=0)
        preprocessed_embeddings = self.duration_predictor_request.get_tensor("embeddings").data[:]

        if non_empty_symbols is not None:
            duration = duration[:, :non_empty_symbols]
            preprocessed_embeddings = preprocessed_embeddings[:, :non_empty_symbols]
        indexes = self.build_index(duration, preprocessed_embeddings)
        if self.verbose:
            log.debug("Index: {0}, duration: {1}, embeddings: {2}, non_empty_symbols: {3}"
                      .format(indexes.shape, duration.shape, preprocessed_embeddings.shape, non_empty_symbols))

        return self.gather(preprocessed_embeddings, 1, indexes)

    def infer_mel(self, aligned_emb, non_empty_symbols, speaker_embedding=None):
        if self.is_attention:
            data_mask = self.sequence_mask(np.array([[non_empty_symbols]]), aligned_emb.shape[1])
            pos_mask = self.pos_mask[:, :, :aligned_emb.shape[1], :aligned_emb.shape[1]]
            inputs = {"data": aligned_emb,
                      "data_mask": data_mask,
                      "pos_mask": pos_mask}
            if speaker_embedding is not None:
                inputs["speaker_embedding"] = np.array(speaker_embedding)
            self.forward_request.infer(inputs)
        else:
            self.forward_request.infer(inputs={"data": aligned_emb})
        return self.forward_request.get_tensor('mel').data[:, :non_empty_symbols]

    def find_optimal_delimiters_position(self, sequence, delimiters, idx, window=20):
        res = {d: -1 for d in delimiters}
        for i in range(max(0, idx - window), idx):
            if sequence[i] in delimiters:
                res[sequence[i]] = i + 1
        return res

    def forward_duration_prediction_by_delimiters(self, text, speaker_embedding, alpha):
        sequence = self.seq_to_indexes(text)
        seq_len = len(sequence)
        outputs = []

        if seq_len <= self.duration_len:
            non_empty_symbols = len(sequence) + min(1, self.duration_len - seq_len)
            sequence = sequence + [_symbol_to_id[' ']] * (self.duration_len - seq_len)
            sequence = np.array(sequence)
            sequence = np.expand_dims(sequence, axis=0)
            outputs.append(self.infer_duration(sequence, speaker_embedding, alpha, non_empty_symbols=non_empty_symbols))
        else:
            punctuation = '.!?,;: '
            delimiters = [_symbol_to_id[p] for p in punctuation]

            start_idx = 0
            while start_idx < seq_len:
                if start_idx + self.duration_len < seq_len:
                    positions = self.find_optimal_delimiters_position(sequence, delimiters,
                                                                      start_idx + self.duration_len,
                                                                      window=self.duration_len//10)
                else:
                    positions = {delimiters[0]: seq_len}
                edge = -1
                for d in delimiters:
                    if positions[d] > 0:
                        edge = positions[d]
                        break
                if edge < 0:
                    raise Exception("Bad delimiter position {0} for sequence with length {1}".format(edge, seq_len))

                sub_sequence = sequence[start_idx:edge]
                non_empty_symbols = len(sub_sequence) + min(1, self.duration_len - len(sub_sequence))
                sub_sequence += [_symbol_to_id[' ']] * (self.duration_len - len(sub_sequence))
                sub_sequence = np.array(sub_sequence)
                sub_sequence = np.expand_dims(sub_sequence, axis=0)
                outputs.append(self.infer_duration(sub_sequence, speaker_embedding, alpha, non_empty_symbols=non_empty_symbols))
                start_idx = edge

        aligned_emb = np.concatenate(outputs, axis=1)
        return aligned_emb

    def forward(self, text, alpha=1.0, speaker_id=19, speaker_emb=None):
        speaker_embedding = None
        if self.is_multi_speaker:
            if speaker_emb is not None:
                speaker_embedding = speaker_emb
            else:
                speaker_embedding = [self.speaker_embeddings[speaker_id, :]]

        aligned_emb = self.forward_duration_prediction_by_delimiters(text, speaker_embedding, alpha)

        mels = []
        start_idx = 0
        end_idx = 0
        while start_idx < aligned_emb.shape[1] and end_idx < aligned_emb.shape[1]:
            end_idx = min(start_idx + self.forward_len, aligned_emb.shape[1])
            sub_aligned_emb = aligned_emb[:, start_idx:end_idx, :]
            if sub_aligned_emb.shape[1] < self.forward_len:
                sub_aligned_emb = np.pad(sub_aligned_emb,
                                         ((0, 0), (0, self.forward_len - sub_aligned_emb.shape[1]), (0, 0)),
                                         'constant', constant_values=0)
            if self.verbose:
                log.debug("SAEmb shape: {0}".format(sub_aligned_emb.shape))
            mel = self.infer_mel(sub_aligned_emb, end_idx - start_idx, speaker_embedding)
            mels.append(np.copy(mel))
            start_idx += self.forward_len

        res = np.concatenate(mels, axis=1)
        if self.verbose:
            log.debug("MEL shape :{0}".format(res.shape))

        return res

    def get_speaker_embeddings(self):
        if self.is_multi_speaker:
            return self.speaker_embeddings
        return None

    def get_pca_speaker_embedding(self, gender, alpha):
        if not self.is_multi_speaker:
            return None

        emb = self.male_embeddings if gender == "Male" else self.female_embeddings
        pca = PCA()
        projection = pca.build(emb)
        x1 = min(projection)
        x2 = max(projection)
        pca_component = x1 + alpha * (x2 - x1)
        emb = pca.iproject(np.array([pca_component]))
        return emb

    def init_speaker_information(self):
        self.male_idx = [2, 3, 7, 11, 12, 15, 16, 19, 20, 21, 25, 26, 27, 29, 32, 33, 34, 35, 36, 38]
        self.female_idx = [0, 1, 4, 5, 6, 8, 9, 10, 13, 14, 17, 18, 22, 23, 24, 28, 30, 31, 37, 39]
        self.speaker_embeddings = np.array([[-0.4327550530433655, -0.5420686602592468],
                                   [-0.5264465808868408, -0.6281864643096924],
                                   [0.15513141453266144, 0.7856010794639587],
                                   [0.3424123525619507, 0.8129010200500488],
                                   [-0.6081429719924927, -0.6511518359184265],
                                   [-0.49752333760261536, -0.8568740487098694],
                                   [-0.005007751286029816, -1.3364707231521606],
                                   [0.14275427162647247, 1.121581792831421],
                                   [-0.45601722598075867, -0.9648892283439636],
                                   [-0.26137179136276245, -1.1388417482376099],
                                   [0.12628738582134247, -1.149622917175293],
                                   [0.34105026721954346, 1.0184416770935059],
                                   [0.3222722113132477, 1.070836067199707],
                                   [-0.2694351375102997, -0.9980007410049438],
                                   [-0.11780811846256256, -1.0476068258285522],
                                   [0.2472933977842331, 1.1816325187683105],
                                   [0.04263993725180626, 1.4357256889343262],
                                   [0.05275965854525566, -1.0010212659835815],
                                   [-0.17100927233695984, -1.1538763046264648],
                                   [0.09288709610700607, 1.296027660369873],
                                   [0.13041983544826508, 1.1497610807418823],
                                   [0.11197542399168015, 1.0537633895874023],
                                   [-0.13089995086193085, -1.2036861181259155],
                                   [0.055261872708797455, -1.338423728942871],
                                   [0.20335668325424194, -1.2085381746292114],
                                   [-0.038247253745794296, 1.268439769744873],
                                   [-0.11069679260253906, 1.050403356552124],
                                   [-0.19113299250602722, 1.0872247219085693],
                                   [0.17568981647491455, -1.247299075126648],
                                   [-0.34791627526283264, 1.0054986476898193],
                                   [0.2401651293039322, -1.1724580526351929],
                                   [0.30263951420783997, -1.043319582939148],
                                   [-0.3040805160999298, 1.1061657667160034],
                                   [-0.27853792905807495, 1.145222544670105],
                                   [-0.49230968952178955, 0.9106340408325195],
                                   [-0.45115727186203003, 0.9025603532791138],
                                   [-0.49153658747673035, 0.7804651260375977],
                                   [0.253637433052063, -1.014277696609497],
                                   [-0.48516881465911865, 0.6745203137397766],
                                   [0.3036082983016968, -0.8406648635864258]])
        mask = np.array([True if i in self.male_idx else False for i in range(self.speaker_embeddings.shape[0])])
        self.male_embeddings = self.speaker_embeddings[mask, :]
        mask = np.array([True if i in self.female_idx else False for i in range(self.speaker_embeddings.shape[0])])
        self.female_embeddings = self.speaker_embeddings[mask, :]
