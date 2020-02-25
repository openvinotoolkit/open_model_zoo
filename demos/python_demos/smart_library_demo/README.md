# Smart Library Demo

The application is a demo of automated “smart library”. 
It involves the registration of the reader; authorization of the reader through face recognition; 
receiving and returning books by recognizing QR codes generated for each book in the library. 
The following pretrained models can be used:

* `face-detection-retail-0004`, to detect faces and predict their bounding boxes;
* `landmarks-regression-retail-0009`, to predict face keypoints;
* `face-reidentification-retail-0095`, to recognize readers.

For more information about the pre-trained models, refer to the [model documentation](../../../models/intel/index.md).

### How it works

The application is started from the command line.
It reads video stream frame-by-frame from a web-camera device and performs independent analysis
of each frame. To make predictions the application uses 3 models. An input frame is processed by
the face detection model to detect bounding boxes. Then face keypoints
are predicted by the facial landmarks regression model. Keypoints are used
to align the face and match it with faces in the database. 
To register in the library press `r` on keyboard. After the reader has been registered, he will be authorized 
through face recognition. Registration allows receiving and returning books by recognizing QR codes for 
each book. To make book recognition press `b` on the keyboard. Also application provides some 
extra statistics in console, like a list of registered readers, full list of books in the library,
history of borrowing books. To change the information in console press `f` on the keyboard. To exit press `q`.

### Installation and dependencies

To install all the required Python modules you can use:\
'''
pip install -r requirements.txt 
'''

### Creating QR-codes for books

The following files are used to create QR-codes for books:
* `library.json` file contains information about books in the library. 
* `createQRCodes.py` script generates QR-codes for each book in `library.json` file.

``` sh
usage: createQRCodes.py [-h] [-i LIB] [-o OUT]

optional arguments:
  -h, --help  show this help message and exit
  -i LIB      unput `.json` file with info
  -o OUT      directory to save generated QR-codes
```

### Running the demo

Running the application with the `-h` option or without
any arguments yield the following message:

``` sh
python ./smart_library_demo.py -h

usage: smart_library_demo.py [-h] -fr FREC -m_rd RDMODEL -fd FDDET -m_fd
                             FDMODEL -lm LMDET -m_lm LMMODEL [-w_rd RDWIDTH]
                             [-h_rd RDHEIGHT] [-t_rd RDTHRESHOLD]
                             [-w_fd FDWIDTH] [-h_fd FDHEIGHT]
                             [-t_fd FDTHRESHOLD] [-w_lm LMWIDTH]
                             [-h_lm LMHEIGHT] [-br BR] [-lib LIB] [-w WEB]
                             [--no_show]

Smart Library Sample

optional arguments:
  -h, --help         show this help message and exit
  -fr FREC           Required. Type of recognizer. Available DNN face
                     recognizer - DNNfr
  -m_rd RDMODEL      Required. Path to .xml file
  -fd FDDET          Required. Type of detector. Available DNN face detector -
                     DNNfd
  -m_fd FDMODEL      Required. Path to .xml file
  -lm LMDET          Required. Type of detector. Available DNN landmarks
                     regression - DNNlm
  -m_lm LMMODEL      Required. Path to .xml file
  -w_rd RDWIDTH      Optional. Image width to resize
  -h_rd RDHEIGHT     Optional. Image height to resize
  -t_rd RDTHRESHOLD  Optional. Probability threshold for face detections.
  -w_fd FDWIDTH      Optional. Image width to resize
  -h_fd FDHEIGHT     Optional. Image height to resize
  -t_fd FDTHRESHOLD  Optional. Probability threshold for face detections.
  -w_lm LMWIDTH      Optional. Image width to resize
  -h_lm LMHEIGHT     Optional. Image height to resize
  -br BR             Optional. Type - QR
  -lib LIB           Optional. Path to library.
  -w WEB             Optional. Specify index of web-camera to open. Default is
                     0
  --no_show          Optional. Don't show output
```

Example of a valid command line to run the application:

Linux (`sh`, `bash`, ...) (assuming OpenVINO installed in `/opt/intel/openvino`):

``` sh
# Set up the environment
source /opt/intel/openvino/bin/setupvars.sh

python ./smart_library_demo.py \
-fr=DNNfr \
-m_rd=<path_to_model>/face-reidentification-retail-0095.xml \
-fd=DNNfd -m_fd=<path_to_model>/face-detection-retail-0004.xml  \
-lm=DNNlm \
-m_lm=<path_to_model>/landmarks-regression-retail-0009.xml \
```

Windows (`cmd`, `powershell`) (assuming OpenVINO installed in `C:/Program Files (x86)/IntelSWTools/openvino/`):

``` powershell
# Set up the environment
call C:/Program Files (x86)/IntelSWTools/openvino_2019.3.334/bin/setupvars.bat

python smart_library_demo.py  -fr=DNNfr -m_rd=<path_to_model>/face-reidentification-retail-0095.xml 
                              -fd=DNNfd -m_fd=<path_to_model>/face-detection-retail-0004.xml 
                              -lm=DNNlm -m_lm=<path_to_model>/landmarks-regression-retail-0009.xml
```
Notice that the custom networks should be converted to the
Inference Engine format (*.xml + *bin) first. To do this use the
[Model Optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) tool.

### Demo output

The demo uses OpenCV window to display the resulting video frame and detections.
It outputs logs to the terminal.

## See also

* [Using Inference Engine Demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
