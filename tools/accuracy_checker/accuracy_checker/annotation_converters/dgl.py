import re

from ._reid_common import check_dirs, read_directory
from .format_converter import GraphFileBasedAnnotationConverter, ConverterReturn
from ..representation import ClassificationAnnotation
from pathlib import Path
import dgl

MARS_IMAGE_PATTERN = re.compile(r'([\d]+)C(\d)')


class DGLConverter(GraphFileBasedAnnotationConverter):
    __provider__ = 'dgl'
    annotation_types = (ClassificationAnnotation, )

    def convert(self, check_content=False, **kwargs):
        print('run convert')
        graph = dgl.data.utils.load_graphs(Path(self.graph_path).__str__())
        g = graph[0][0]

        labels = g.ndata["label"]

        annotation = [
            ClassificationAnnotation(label=labels)
        ]

        return ConverterReturn(annotation, {'labels': labels}, None)
