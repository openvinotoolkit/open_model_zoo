"""
 Copyright (c) 2021 Intel Corporation

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

from collections import defaultdict

import cv2
import numpy as np

from .model import Model


class TextRecognition(Model):
    def __init__(self, ie, model_path, alphabet, bandwidth):
        super().__init__(ie, model_path)

        self.image_blob_name = self.prepare_inputs()
        self.output_blob_name = self.prepare_outputs()

        self.alphabet = alphabet
        self.blank_label = self.alphabet.index('#')

        if bandwidth == 0:
            self.decoder = CTCGreedySearchDecoder(self.blank_label)
        else:
            self.decoder = BeamSearchDecoder(self.blank_label, bandwidth)

    def prepare_inputs(self):
        if len(self.net.input_info) != 1:
            raise RuntimeError("The model topology supposes only 1 input layer")

        image_blob_name = next(iter(self.net.input_info))
        input_size = self.net.input_info[image_blob_name].input_data.shape

        if len(input_size) == 4 and input_size[1] == 1:
            self.h, self.w = input_size[2:]
        else:
            raise RuntimeError("Grayscale 4-dimensional model's input is expected")

        return image_blob_name

    def prepare_outputs(self):
        if len(self.net.outputs) == 1:
            output_blob_name = next(iter(self.net.outputs))
        else:
            output_blob_name = [name for name in self.net.outputs if name.startswith('logits')][0]

        for _, blob in self.net.input_info.items():
            if len(blob.input_data.shape) != 3:
                raise RuntimeError("Unexpected output blob shape. Only 3D output blobs are supported")

        return output_blob_name

    def preprocess(self, inputs):
        image = cv2.cvtColor(inputs, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(image, (self.w, self.h))
        dict_inputs = {self.image_blob_name: resized_image}
        return dict_inputs, None

    def postprocess(self, outputs, meta):
        output = outputs[self.output_blob_name]
        seq = self.decoder.process(output)
        return ''.join(self.alphabet[char] for char in seq)


class CTCGreedySearchDecoder:
    def __init__(self, blank_label):
        self.blank_label = blank_label

    def process(self, output):
        preds_index = np.argmax(output, 2).transpose(1, 0)
        return self.decode(preds_index[0], self.blank_label)

    @staticmethod
    def decode(prob_index, blank_id):
        """
         Decode given output probabilities to sequence of labels.
        Arguments:
            prob_index: The max index along the probabilities dimension.
            blank_id (int): Index of the CTC blank label.
        Returns the output label sequence.
        """
        selected_index = []
        for i, _ in enumerate(prob_index):
            # removing repeated characters and blank.
            if prob_index[i] != blank_id and (not (i > blank_id and prob_index[i - 1] == prob_index[i])):
                selected_index.append(prob_index[i])
        return selected_index


class BeamSearchDecoder:
    def __init__(self, blank_label, bandwidth):
        self.blank_label = blank_label
        self.bandwidth = bandwidth
        self.softmaxed_probabilities = False

    def process(self, output):
        output = np.swapaxes(output, 0, 1)
        if self.softmaxed_probabilities:
            output = np.log(output)
        return self.decode(output[0], self.bandwidth, self.blank_label)

    @staticmethod
    def decode(probabilities, beam_size=10, blank_id=None):
        """
         Decode given output probabilities to sequence of labels.
        Arguments:
            probabilities: The output log probabilities for each time step.
            Should be an array of shape (time x output dim).
            beam_size (int): Size of the beam to use during decoding.
            blank_id (int): Index of the CTC blank label.
        Returns the output label sequence.
        """
        def make_new_beam():
            return defaultdict(lambda: (-np.inf, -np.inf))

        def log_sum_exp(*args):
            if all(a == -np.inf for a in args):
                return -np.inf
            a_max = np.max(args)
            lsp = np.log(sum(np.exp(a - a_max) for a in args))

            return a_max + lsp

        times, symbols = probabilities.shape
        # Initialize the beam with the empty sequence, a probability of 1 for ending in blank
        # and zero for ending in non-blank (in log space).
        beam = [(tuple(), (0.0, -np.inf))]

        for time in range(times):
            # A default dictionary to store the next step candidates.
            next_beam = make_new_beam()

            for symbol_id in range(symbols):
                current_prob = probabilities[time, symbol_id]

                for prefix, (prob_blank, prob_non_blank) in beam:
                    # If propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
                    if symbol_id == blank_id:
                        next_prob_blank, next_prob_non_blank = next_beam[prefix]
                        next_prob_blank = log_sum_exp(
                            next_prob_blank, prob_blank + current_prob, prob_non_blank + current_prob
                        )
                        next_beam[prefix] = (next_prob_blank, next_prob_non_blank)
                        continue
                    # Extend the prefix by the new character symbol and add it to the beam.
                    # Only the probability of not ending in blank gets updated.
                    end_t = prefix[-1] if prefix else None
                    next_prefix = prefix + (symbol_id,)
                    next_prob_blank, next_prob_non_blank = next_beam[next_prefix]
                    if symbol_id != end_t:
                        next_prob_non_blank = log_sum_exp(
                            next_prob_non_blank, prob_blank + current_prob, prob_non_blank + current_prob
                        )
                    else:
                        # Don't include the previous probability of not ending in blank (prob_non_blank) if symbol
                        #  is repeated at the end. The CTC algorithm merges characters not separated by a blank.
                        next_prob_non_blank = log_sum_exp(next_prob_non_blank, prob_blank + current_prob)

                    next_beam[next_prefix] = (next_prob_blank, next_prob_non_blank)
                    # If symbol is repeated at the end also update the unchanged prefix. This is the merging case.
                    if symbol_id == end_t:
                        next_prob_blank, next_prob_non_blank = next_beam[prefix]
                        next_prob_non_blank = log_sum_exp(next_prob_non_blank, prob_non_blank + current_prob)
                        next_beam[prefix] = (next_prob_blank, next_prob_non_blank)

            beam = sorted(next_beam.items(), key=lambda x: log_sum_exp(*x[1]), reverse=True)[:beam_size]

        return beam[0][0]
