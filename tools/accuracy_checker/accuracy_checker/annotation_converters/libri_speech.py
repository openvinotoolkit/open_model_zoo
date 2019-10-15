###############################################
#
# Create by John Feng
# Contact: john.feng@intel.com
#
###############################################

from ..config import PathField
from ..representation import CharacterRecognitionAnnotation
from ..utils import read_txt, check_file_existence
from .format_converter import FileBasedAnnotationConverter, ConverterReturn

class LibriSpeechFormatConverter(FileBasedAnnotationConverter):
    __provider__ = 'libri_speech'

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

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        content_errors = None
        if check_content:
            content_errors = []
            self.images_dir = self.images_dir or self.annotation_file.parent

        label_files = []

        for files_in_dir in self.data_dir.iterdir():
            str_file_in_dir = str(files_in_dir)
            if '.txt' in str_file_in_dir:
                label_files.append(str_file_in_dir)

        num_iterations = len(label_files)

        for _id, _file in enumerate(label_files):
            label = read_txt(_file)[0].lower()
            audio_name = _file[:-3] + 'wav'
            annotations.append(CharacterRecognitionAnnotation(audio_name, label))
            if check_content:
                if not check_file_existence(self.images_dir / audio_name):
                    content_errors.append('{}: does not exist'.format(audio_name))
            if progress_callback is not None and _id % progress_interval:
                progress_callback(_id / num_iterations * 100)

        label_map = {ind: str(key) for ind, key in enumerate(self.supported_symbols)}
        meta = {'label_map': label_map, 'blank_label': len(label_map)}

        return ConverterReturn(annotations, meta, content_errors)
