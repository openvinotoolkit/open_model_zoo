#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import abc

import numpy as np


class SeqPipelineStage(abc.ABC):
    """
    One stage of a streaming data pipeline.

    Each data piece (in or out) can be numpy.ndarray or None (empty piece).
    Concatenated (along axis=0) outputs must depend only on concatenated inputs,
    and must be invariant to different ways of representing this input.

    In case your output depends on whole input sequence (e.g. BiLSTM), accumulate input
    sequence into a buffer and process the whole sequence in the end (when finish=True).
    """
    @abc.abstractmethod
    def process_data(self, data, finish=False):
        """
            Returns:
        numpy.ndarray (any stage) or other type (for the last stage only), processing result
          OR
        None (empty data)
        """
        pass


class SeqPipeline(SeqPipelineStage):
    def __init__(self, stages):
        self._stages = []
        for stage in stages:
            self.add_stage(stage)

    def add_stage(self, stage):
        self._stages.append(stage)

    def process_data(self, data, finish=False):
        for stage in self._stages:
            data = stage.process_data(data, finish=finish)
        return data


class BlockedSeqPipelineStage(SeqPipelineStage):
    """
    Streaming pipeline that provides common methods for processing data in blocks of some fixed size.
    """
    def __init__(self, block_len, context_len, left_padding_len, right_padding_len, padding_shape, cut_alignment):
        assert block_len > 0 and context_len >= 0 and left_padding_len >= 0 and right_padding_len >= 0
        self._block_len = block_len
        self._context_len = context_len
        self._left_padding_len = left_padding_len
        self._right_padding_len = right_padding_len
        self._padding_shape = padding_shape
        self._cut_alignment = cut_alignment
        self._reset_state()

    def _reset_state(self):
        self._buffer = None  # None for buffer without left padding, [] for buffer with empty left padding
        self._buffer_len = 0

    def _finalize_and_reset_state(self):
        """
        Finalize state with no additional data, and return the output that results from finalization.
        This method is overridden in CTC decoder, which can update its output because of finalization.

            Return
        Same output as in process_data()
        """
        self._reset_state()
        return None

    def process_data(self, data, finish=False):
        """
            Args:
        data (numpy.ndarray or None), new data to ba concatenated with the older data along axis=0, None for no new data
        finish (bool), set to True for the last segment of data to finalize processing and flush buffers

            Returns:
        numpy.ndarray or None, output data (any data type can be returned for the last stage in pipeline)
        """
        # === Accept new data and prepare it for processing if there's enough data ===
        if data is not None:
            if self._buffer is None:
                self._buffer = []
                if self._left_padding_len != 0:
                    self._buffer.append(np.zeros((self._left_padding_len, *self._padding_shape), dtype=data.dtype))
                self._buffer_len = self._left_padding_len
            self._buffer.append(data)
            self._buffer_len += data.shape[0]

        if finish:
            if self._buffer is None:
                # Finalizing without input data at all -- no need to call _reset_state() in this case
                return None
            align_right_len = (-(self._buffer_len + self._right_padding_len - self._context_len)) % self._block_len
            pad_right_len = self._right_padding_len + align_right_len
            if pad_right_len > 0:
                self._buffer.append(np.zeros((pad_right_len, *self._padding_shape), dtype=self._buffer[0].dtype))
                self._buffer_len += pad_right_len
        else:
            align_right_len = 0

        if self._buffer_len < self._block_len + self._context_len:
            if finish:
                # Have some data, but not enough data+padding for another block -- dismissing buffered context
                return self._finalize_and_reset_state()
            # No new data to output, returning None for empty segment
            return None
        # Now we're guaranteed to have self._buffer is not None and self._buffer_len > 0 because of the last "if"
        assert self._buffer is not None and self._buffer_len > 0

        buffer = np.concatenate(self._buffer, axis=0)
        self._buffer = [buffer]
        self._buffer_len = buffer.shape[0]

        # === Loop over blocks ===
        # variables accepted from prev.stage: buffer, finalize
        processed, buffer_skip_len = self._process_blocks(buffer, finish=finish)
        if finish:
           self._reset_state()
        else:
            # start_pos contains its value for the last iteration of the loop
            buffer = buffer[buffer_skip_len:].copy()
            self._buffer = [buffer]
            self._buffer_len = self._buffer[0].shape[0]

        # === Postprocess ===
        # variables accepted from prev.stage: processed, align_right_len
        if self._cut_alignment and finish and align_right_len > 0:
            # Crop alignment padding from the last block.
            processed[-1] = processed[-1][:-align_right_len]

        return self._combine_output(processed)

    def _process_blocks(self, buffer, finish=False):
        """
        Process buffer with data enough for one or more blocks

          Args:
        buffer (numpy.ndarray), buffer is guaranteed to contain data for 1 or more blocks
            (buffer.shape[0]>=self._block_len+self._context_len)
        finish (bool)

          Return:
        list of numpy.ndarray, to be concatenated along axis=0 outside this method
        """
        assert buffer.shape[0] >= self._block_len + self._context_len
        processed = []
        for start_pos in range(self._context_len, buffer.shape[0] - self._block_len + 1, self._block_len):
            block = buffer[start_pos - self._context_len:start_pos + self._block_len]
            processed.append(self._process_block(block, finish=finish and start_pos + self._block_len >= buffer.shape[0]))
        assert not self._cut_alignment or processed[-1].shape[0] == self._block_len, "Networks with stride != 1 are not supported"
        # Here start_pos is its value on the last iteration of the loop
        buffer_skip_len = start_pos + self._block_len - self._context_len
        return processed, buffer_skip_len

    def _process_block(self, block, finish=False):
        raise NotImplementedError("_process_block() should have been implemented in case inherited _process_blocks() is used")

    def _combine_output(self, processed_list):
        return np.concatenate(processed_list, axis=0)
