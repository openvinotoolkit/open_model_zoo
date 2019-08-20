from .format_converter import FileBasedAnnotationConverter, ConverterReturn
from ..representation import DetectionAnnotation
from ..topology_types import ObjectDetection
from ..utils import read_xml, check_file_existence
from ..config import PathField, ConfigError, BoolField


class CVATForObjectDetection(FileBasedAnnotationConverter):
    __provider__ = 'cvat_object_detection'
    annotation_types = (DetectionAnnotation, )
    topology_types = (ObjectDetection, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'images_dir': PathField(
                is_directory=True, optional=True,
                description='path to dataset images, used only for content existence check'
            ),
            'has_background': BoolField(optional=True, default=True, description='Dataset has background label or not')
        })
        return parameters

    def configure(self):
        super().configure()
        self.has_background = self.get_value_from_config('has_background')
        self.images_dir = self.get_value_from_config('images_dir') or self.annotation_file.parent

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = read_xml(self.annotation_file)
        meta = annotation.find('meta')
        size = int(meta.find('task').find('size').text)
        labels = [label.find('name').text for label in meta.iter('label') if label.find('name').text]
        if not labels:
            raise ConfigError('annotation file does not contains labels')
        if self.has_background:
            labels = ['background'] + labels
        label_to_id = {label: idx for idx, label in enumerate(labels)}

        annotations = []
        content_errors = None if not check_content else []
        for image_id, image in enumerate(annotation.iter('image')):
            identifier = image.attrib['name'].split('/')[-1]
            if check_content:
                if not check_file_existence(self.images_dir / identifier):
                    content_errors.append('{}: does not exist'.format(self.images_dir / identifier))
            x_mins, y_mins, x_maxs, y_maxs, labels_ids, difficult = [], [], [], [], [], []
            for bbox_id, bbox in enumerate(image):
                if 'label' not in bbox.attrib.keys() or bbox.attrib['label'] not in label_to_id:
                    continue
                labels_ids.append(label_to_id[bbox.attrib['label']])
                x_mins.append(float(bbox.attrib['xtl']))
                y_mins.append(float(bbox.attrib['ytl']))
                x_maxs.append(float(bbox.attrib['xbr'])),
                y_maxs.append(float(bbox.attrib['ybr']))
                if 'occluded' in bbox.attrib and int(bbox.attrib['occluded']):
                    difficult.append(bbox_id)
            detection_annotation = DetectionAnnotation(identifier, labels_ids, x_mins, y_mins, x_maxs, y_maxs)
            detection_annotation.metadata['difficult_boxes'] = difficult
            annotations.append(detection_annotation)
            if progress_callback is not None and image_id % progress_interval == 0:
                progress_callback(image_id * 100 / size)

        return ConverterReturn(annotations, self.generate_meta(label_to_id), content_errors)

    def generate_meta(self, values_mapping):
        meta = {'label_map': {value: key for key, value in values_mapping.items()}}
        if self.has_background:
            meta['background_label'] = 0

        return meta
