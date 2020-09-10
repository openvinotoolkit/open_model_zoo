# Data Readers

Data Reader is a function for reading input data.
You can use 2 ways to set data_reader for dataset:
* Define reader as a string.

```yml
reader: opencv_imread
```

* Define reader as a dictionary, using `type:` for setting reader name. This approach gives opportunity to set additional parameters for adapter if it is required.

```yml
reader:
  type: opencv_imread
  reading_flag: gray
```

In case, when you have model with several inputs which should use data stored in different format (e. g. images and json) you can use `combine_reader`.
`combine_reader` allows specify reading scheme depends on file names. It use parameter `scheme` for describing reading approaches as dictionary where keys are regular expressions for file names, values are reader_name.

```yml
reader:
  type: combine_reader
  scheme:
    *.json: json_reader
    *.jpeg: opencv_imread
```

AccuracyChecker supports following list of data readers:
* `opencv_imread` - read images using OpenCV library. Default color space is BGR.
   * `reading_flag` - (Optional) flag which specifies the way image should be read: `color` - default, loads color image, `gray` - loads image in grayscale mode, `unchanged` - loads image as such including alpha channel.
* `pillow_imread` - read images using Pillow library. Default color space is RGB.
* `scipy_imread` - read images using similar approach as in `scipy.misc.imread`
```
Note: since 1.3.0 version the image processing module is not a part of scipy library. This reader does not use scipy anymore.
```
* `tf_imread`- read images using TensorFlow. Default color space is RGB. Requires TensorFlow installation.
* `opencv_capture` - read frames from video using OpenCV.
* `json_reader` - read value from json file.
  * `key` - key for reading from stored in json dictionary.
* `annotation_features_extractor` - read features from annotation.
  * `features` - list of features. All features should be fields of annotation representation.
* `numpy_reader` - read numpy dumped files (npy or npz formats are supported for reading)
  * `keys` - comma-separated list of model input names
  * `separator` - separator symbol between input identifier and file identifier
  * `id_sep` - separator symbol between input name and record number in input identifier
  * `block` - block mode (batch - oriented). In this mode reader returns whole variable.
* `numpy_txt_reader`- read data stored in text format to numpy array.
* `numpy_dict_reader` - read and unpack dictionaries saved in numpy files.
* `nifti_reader` - read NifTI data format
  * `channels_first` - allows read nifti files and transpose in order where channels first (Optional, default False)
* `wav_reader` - read WAV file into NumPy array. Also gets the samplerate.
* `dicom_reader` - read images stored in DICOM format.
* `pickle_reader` - read data stored in pickle file. Supported formats of pickle content:
  1. numeric data array
  2. numeric data array + metadata stored in dictionary
