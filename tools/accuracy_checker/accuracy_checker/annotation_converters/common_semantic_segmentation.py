from ..config import PathField, StringField
from ..logging import warning
from .format_converter import BaseFormatConverter, ConverterReturn, verify_label_map
from ..representation.segmentation_representation import LOADERS_MAPPING
from ..representation import SegmentationAnnotation
from ..utils import read_json


class CommonSegmentationConverter(BaseFormatConverter):
    __provider__ = 'common_semantic_segmentation'

    @classmethod
    def parameters(cls):
        configuration_parameters = super().parameters()
        configuration_parameters.update(
            {
                'images_dir': PathField(description='path to input images directory', is_directory=True),
                'masks_dir': PathField(description='path to gt masks directory', is_directory=True),
                'image_prefix': StringField(optional=True, default='', description='prefix for images'),
                'mask_prefix': StringField(optional=True, default='', description='prefix for gt masks'),
                'image_postfix': StringField(optional=True, default='.png', description='prefix for images'),
                'mask_postfix': StringField(optional=True, default='.png', description='prefix for gt masks'),
                'mask_loader': StringField(
                    optional=True, choices=LOADERS_MAPPING,
                    description='reader for gt masks. Supported: {}'.format(', '.join(LOADERS_MAPPING)),
                    default='pillow'
                ),
                'dataset_meta_file': PathField(
                    description='path to json file with dataset meta (e.g. label_map, color_encoding', optional=True
                )
            }
        )
        return configuration_parameters

    def configure(self):
        self.images_dir = self.get_value_from_config('images_dir')
        self.masks_dir = self.get_value_from_config('masks_dir')
        self.images_prefix = self.get_value_from_config('image_prefix')
        self.images_postfix = self.get_value_from_config('image_postfix')
        self.mask_prefix = self.get_value_from_config('mask_prefix')
        self.mask_postfix = self.get_value_from_config('mask_postfix')
        self.mask_loader = LOADERS_MAPPING[self.get_value_from_config('mask_loader')]
        self.dataset_meta = self.get_value_from_config('dataset_meta_file')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotations = []
        mask_name = '{prefix}{base}{postfix}'.format(
            prefix=self.mask_prefix, base='{base}', postfix=self.mask_postfix
        )
        image_pattern = '*'
        if self.images_prefix:
            image_pattern = self.images_prefix + image_pattern
        if self.images_postfix:
            image_pattern = image_pattern + self.images_postfix
        images_list = list(self.images_dir.glob(image_pattern))
        num_iterations = len(images_list)
        content_errors = None if not check_content else []
        for idx, image in enumerate(images_list):
            base_name = image.name
            identifier = base_name
            if self.images_prefix:
                base_name = base_name.split(self.images_prefix)[-1]
            if self.images_postfix:
                base_name = base_name.split(self.images_postfix)[0]

            mask_file = self.masks_dir / mask_name.format(base=base_name)
            if not mask_file.exists():
                content_errors.append('{}: does not exist'.format(mask_file))

            annotations.append(SegmentationAnnotation(identifier, mask_file.name, mask_loader=self.mask_loader))
            if progress_callback is not None and idx % progress_interval == 0:
                progress_callback(idx / num_iterations * 100)

        dataset_meta = None
        if self.dataset_meta:
            dataset_meta = read_json(self.dataset_meta)
            if 'label_map' not in dataset_meta:
                if 'labels' in dataset_meta:
                    dataset_meta['label_map'] = dict(enumerate(dataset_meta['labels']))
                else:
                    warning("Information about dataset labels is provided. Please provide it for metric calculation.")
            else:
                dataset_meta['label_map'] = verify_label_map(dataset_meta['label_map'])

        return ConverterReturn(annotations, dataset_meta, content_errors)
