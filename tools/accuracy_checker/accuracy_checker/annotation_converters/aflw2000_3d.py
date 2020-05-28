import numpy as np
from ..config import ConfigError
from ..representation import FacialLandmarks3DAnnotation
from .format_converter import DirectoryBasedAnnotationConverter, ConverterReturn

try:
    import scipy.io as scipy_io
except ImportError:
    scipy_io = None


class AFLW20003DConverter(DirectoryBasedAnnotationConverter):
    __provider__ = 'aflw2000_3d'

    def __init__(self, config=None):
        if scipy_io is None:
            raise ConfigError(
                '{} converter require scipy installation. Please install it before usage.'.format(self.__provider__)
            )
        super().__init__(config)

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        images_list = list(self.data_dir.glob('*.jpg'))
        num_iterations = len(images_list)
        content_errors = [] if check_content else None
        annotations = []
        for img_id, image in enumerate(images_list):
            annotation_file = self.data_dir / image.name.replace('jpg', 'mat')
            if not annotation_file.exists():
                if check_content:
                    content_errors.append('{}: does not exists'.format(str(annotation_file)))
                continue

            image_info = scipy_io.loadmat(str(annotation_file))
            x_values, y_values, z_values = image_info['pt3d_68']
            x_min, y_min = np.min(x_values), np.min(y_values)
            x_max, y_max = np.max(x_values), np.max(y_values)
            annotation = FacialLandmarks3DAnnotation(image.name, x_values, y_values, z_values)
            annotation.metadata['rect'] = [x_min, y_min, x_max, y_max]
            annotation.metadata['left_eye'] = [36, 39]
            annotation.metadata['right_eye'] = [42, 45]
            annotations.append(annotation)
            if progress_callback is not None and img_id % progress_interval:
                progress_callback(img_id / num_iterations * 100)

        return ConverterReturn(annotations, None, content_errors)
