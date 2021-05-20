import numpy as np
import cv2

from ..config import PathField, ConfigError, BoolField
from ..utils import contains_all, get_path, check_file_existence, UnsupportedPackage
from ..representation import DepthEstimationAnnotation
from ..representation.depth_estimation import GTLoader
from .format_converter import BaseFormatConverter, ConverterReturn

try:
    import h5py
except ImportError as import_error:
    h5py = UnsupportedPackage("h5py", import_error.msg)

class NYUDepthV2Converter(BaseFormatConverter):
    __provider__ = 'nyu_depth_v2'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'images_dir': PathField(
                optional=True, is_directory=True, description='path to directory with images', check_exists=False
            ),
            'depth_map_dir': PathField(
                optional=True, is_directory=True, description='path to directory with depth maps', check_exists=False
            ),
            'data_dir': PathField(
                is_directory=True, optional=True,
                description='path to directory with data in original hdf5 format stored'
            ),
            'allow_convert_data': BoolField(
                optional=True, default=False, description="Allows to convert data from hdf5 format"
            )
        })
        return parameters

    def configure(self):
        self.data_dir = self.get_value_from_config('data_dir')
        self.allow_convert_data = self.get_value_from_config('allow_convert_data')
        self.images_dir = self.get_value_from_config('images_dir')
        self.depths_dir = self.get_value_from_config('depth_map_dir')

        if self.allow_convert_data:
            if isinstance(h5py, UnsupportedPackage):
                h5py.raise_error(self.__provider__)
            if self.data_dir is None:
                raise ConfigError('please provide data_dir to convert data from hdf5 format')

            if self.images_dir is None:
                self.images_dir = self.data_dir.parent / 'converted/images'
            if self.depths_dir is None:
                self.depths_dir = self.data_dir.parent / 'converted/depth'

            if not self.images_dir.exists():
                self.images_dir.mkdir(parents=True)
            if not self.depths_dir.exists():
                self.depths_dir.mkdir(parents=True)

        else:
            if not contains_all(self.config, ['images_dir', 'depth_map_dir']):
                raise ConfigError('both images_dir and depth_map_dir should be provided')
            self.images_dir = get_path(self.images_dir, is_directory=True)
            self.depths_dir = get_path(self.depths_dir, is_directory=True)


    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        if self.allow_convert_data:
            images_list = self.convert_data()
        else:
            images_list = list(self.images_dir.glob('*.png'))

        annotations = []
        num_iterations = len(images_list)
        content_errors = [] if check_content else None
        for idx, image_path in enumerate(images_list):
            identifier = image_path.name
            depth_file = identifier.replace('png', 'npy')
            if check_content and not check_file_existence(self.depths_dir / depth_file):
                content_errors.append("{}: does not exist".format(self.depths_dir / depth_file))
            annotations.append(DepthEstimationAnnotation(identifier, depth_file, GTLoader.NUMPY))
            if progress_callback and idx % progress_interval == 0:
                progress_callback(idx * 100 / num_iterations)

        return ConverterReturn(annotations, None, content_errors)

    def convert_data(self):
        images = []
        for h5file in self.data_dir.glob('*.h5'):
            with h5py.File(str(h5file), 'r') as f:
                image = np.transpose(f['rgb'], (1, 2, 0))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                depth = f['depth'][:].astype('float16')
                image_path = self.images_dir / h5file.name.replace('h5', 'png')
                cv2.imwrite(str(image_path), image)
                np.save(str(self.depths_dir / h5file.name.replace('h5', 'npy')), depth)
                images.append(image_path)

        return images
