This demo can work with two types of StarDist models and data:

1. [stardist_he](../../../models/public/stardist_he/stardist_he.md) to detect object on RGB slides of brightfield images.
Example data should be in `.ndpi` format. In example, from http://openslide.cs.cmu.edu/download/openslide-testdata/Hamamatsu/

To run demo with `.ndpi` images, install [OpenSlide for Python](https://pypi.org/project/openslide-python/):
```
pip install openslide-python==1.1.2
apt-get install libopenslide-dev
```

2. [stardist_dsb2018](../../../models/public/stardist_dsb2018/stardist_dsb2018.md) to detect object on single channel slides of fluorescence images.
Example data can be downloaded from https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/PKI_scans/ (tested on `LuCa-7color_Scan1.qptiff`)

To run demo on fluorescence images, install [Bio-Formats for Python](https://pythonhosted.org/python-bioformats/)
```
JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/" pip install python-bioformats==4.0.0
```

(Java 8 should be installed)

