######################################
#
#   John Feng, john.feng@intel.com
#
######################################


from ..config import NumberField
from .preprocessor import Preprocessor

class Create_overlap_windows(Preprocessor):
    __provider__ = 'overlap_creator'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'step': NumberField(
                value_type=int, optional=True, min_value=1, description="Number of step length."
            ),
            'context': NumberField(
                value_type=int, optional=True, min_value=1, description="Number of context length."
            ),
            'input': NumberField(
                value_type=int, optional=True, min_value=1, description="Number of input size."
            )
        })

        return parameters

    def configure(self):
        self.step = self.get_value_from_config('step')
        self.context = self.get_value_from_config('context')
        self.input = self.get_value_from_config('input')

    def process(self, image, annotation_meta=None):
        empty_context = np.zeros((self.context, self.input), dtype=image.data.dtype)
        audio = np.concatenate((empty_context, image.data, empty_context))

        num_strides = len(audio) - (self.context * 2)
        window_size = 2 * self.context + 1

        audio = np.lib.stride_tricks.as_strided(audio,
                                                (num_strides, window_size, self.input),
                                                (audio.strides[0], audio.strides[0], audio.strides[1]),
                                                writeable=False)
        image.data = audio

        return image

class Prepare_audio_package(Preprocessor):
    __provider__ = 'audio_package'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'step': NumberField(
                value_type=int, optional=True, min_value=1, description="Number of step length."
            )
        })

        return parameters

    def configure(self):
        self.step = self.get_value_from_config('step')

    def process(self, image, annotation_meta=None):
        data = image.data

        _length, c, i = data.shape

        _p_pad = _length % self.step
        if _p_pad != 0:
            data = np.pad(data, ((0, self.step - _p_pad), (0, 0), (0, 0)), mode='constant', constant_values=0)

        image.data = np.transpose(data.reshape((-1, self.step, c, i)), (0, 2, 3, 1))

        return image
