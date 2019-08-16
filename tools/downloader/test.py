import os

#
# Test paths from MO args
#
from topologies import ssd300

t = ssd300()
xmlPath, binPath = t.getIR()
xmlPath_fp16, binPath_fp16 = t.getIR(precision='FP16')

for path, name in zip ([t.model, t.config, xmlPath, binPath, xmlPath_fp16, binPath_fp16],
                       ['VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.caffemodel',
                        'deploy.prototxt',
                        'ssd300.xml', 'ssd300.bin',
                        'ssd300.xml', 'ssd300.bin']):
    assert(os.path.exists(path)), path
    assert(os.path.basename(path) == name), name
