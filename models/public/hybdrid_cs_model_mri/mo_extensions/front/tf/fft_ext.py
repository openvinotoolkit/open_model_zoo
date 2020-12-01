from ...ops.fft import FFT
from mo.front.extractor import FrontExtractorOp
from mo.utils.error import Error

class FFT2DFrontExtractor(FrontExtractorOp):
    op = 'FFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'inverse': 0
        }
        FFT.update_node_stat(node, attrs)
        return cls.enabled

class IFFT2DFrontExtractor(FrontExtractorOp):
    op = 'IFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'inverse': 1
        }
        FFT.update_node_stat(node, attrs)
        return cls.enabled
