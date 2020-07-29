from ..config import PathField
from ..representation import FeaturesRegressionAnnotation
from .format_converter import ConverterReturn, BaseFormatConverter, StringField


class FeaturesRegressionConverter(BaseFormatConverter):
    __provider__ = 'feature_regression'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'input_dir': PathField(is_directory=True),
            'reference_dir': PathField(is_directory=True),
            'input_suffix': StringField(optional=True, default='.txt'),
            'reference_suffix': StringField(optional=True, default='.txt')
        })
        return params

    def configure(self):
        self.in_directory = self.get_value_from_config('input_dir')
        self.ref_directory = self.get_value_from_config('reference)dir')
        self.in_suffix = self.get_value_from_config('input_suffix')
        self.ref_suffix = self.get_value_from_config('reference_suffix')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        ref_data_list = list(self.ref_directory.glob('*{}'.format(self.ref_suffix)))
        for ref_file in ref_data_list:
            identifier = ref_file.name.split(self.ref_suffix)[0] + self.in_suffix
            if not (self.in_directory / identifier).exists():
                continue
            annotations.append(FeaturesRegressionAnnotation(identifier, ref_file.name))
        return ConverterReturn(annotations, None, None)
