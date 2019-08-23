from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..utils import read_txt, check_file_existence
from ..representation import SegmentationAnnotation


class CamVidConverter(FileBasedAnnotationConverter):
    __provider__ = 'camvid'
    annotation_type = (SegmentationAnnotation, )
    meta = {
        'label_map': {
            0: 'Sky',
            1: 'Building',
            2: 'Pole',
            3: 'Road',
            4: 'Pavement',
            5: 'Tree',
            6: 'SignSymbol',
            7: 'Fence',
            8: 'Car',
            9: 'Pedestrian',
            10: 'Bicyclist',
            11: 'Unlabelled'
        },
        'background_label': 11,
        'segmentation_colors': (
            (128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (60, 40, 222), (128, 128, 0),
            (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 0)
        )
    }

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_txt(self.annotation_file)
        annotations = []
        content_errors = None if not check_content else []
        num_iterations = len(annotation)
        for line_id, line in enumerate(annotation):
            image_path, gt_path = line.split(' ')
            if check_content:
                if not check_file_existence(image_path):
                    content_errors.append("{}: does not exists".format(image_path))
                if not check_file_existence(gt_path):
                    content_errors.append('{}: does not exist'.format(gt_path))
            identifier = '/'.join(image_path.split('/')[-2:])
            gt_file = '/'.join(gt_path.split('/')[-2:])
            annotations.append(SegmentationAnnotation(identifier, gt_file))
            if progress_callback is not None and line_id % progress_interval == 0:
                progress_callback(line_id * 100 / num_iterations)

        return ConverterReturn(annotations, self.meta, content_errors)
