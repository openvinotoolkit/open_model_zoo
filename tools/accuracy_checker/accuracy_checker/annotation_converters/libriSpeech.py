###############################################
#
# Create by John Feng
# Contact: john.feng@intel.com
# 
###############################################

from ..config import PathField
# from ..representation import SpeechRecognitionAnnotation
from ..representation import CharacterRecognitionAnnotation
from ..utils import read_txt

from .format_converter import BaseFormatConverter

class LibriSpeechFormatConverter(BaseFormatConverter):
    __provider__ = 'libriSpeech'
    
    annotation_types = (CharacterRecognitionAnnotation, )
    supported_symbols = " abcdefghijklmnopqrstuvwxyz'"

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'data_dir': PathField(
                is_directory=True, description="Path to folder, where the audio and label are located."
            ),
        })
        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')

    def convert(self):
        annotation = []
        for file_in_dir in self.data_dir.iterdir():
            str_file_in_dir = str(file_in_dir)
            if '.txt' in str_file_in_dir:
                label = read_txt(str_file_in_dir)[0].lower()
                audio_name = str_file_in_dir[:-3] + 'wav'
                annotation.append(CharacterRecognitionAnnotation(audio_name, label))
    
        label_map = {ind: str(key) for ind, key in enumerate(self.supported_symbols)}

        return annotation, {'label_map': label_map, 'blank_label': len(label_map)}










