# Object Detection Async Demo

This demo showcases Object Detection on Open Model Zoo models with Async API.
Async API usage can improve the overall frame-rate of the application, because
inference and image preprocessing can occur at the same time.

This notebook allows you to select a model and an input video, as well as vary
the number of streams, threads and requests for the inference.

Note: the notebook allows you to upload your own video. It is recommended to
use a short video. If you use a video that is longer than a few minutes, you
can adjust the `JUMP_FRAMES` setting to a larger value to increase inference
speed. With the default setting every tenth frame is analyzed.

Other demo objectives are:

* Using video as input with OpenCV\*
* Visualizing the resulting bounding boxes
* Comparing results and speed of different Open Model Zoo models

See the [Python Object Detection Async Demo](../python/) for more details about
the Async API, and the [Optimization
Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html)
for more information on optimizing models.

## Run this demo on your computer

To run this notebook on your computer, you need to install Python (3.6, 3.7 or 3.8). If you do not have Python yet,
install it from https://www.python.org/downloads/release/python-379/. For Windows, choose the [executable installer
for x86-64](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe) and select the *Add Python to your PATH*
option during installation.

1. Clone the Open Model Zoo repository to your computer with `git clone https://github.com/openvinotoolkit/open_model_zoo.git`

2. Install the required Python packages
   - Open a terminal and go to the demo directory (the directory that contains this README.).
     - On Linux or MacOS, open Terminal. On Windows, open a Command Prompt (type `cmd` in the search Window in the task bar)
     - Use the `cd` command to go to the correct directory. Starting from the main `open_model_zoo` directory, you can type `cd demos/object_detection_demo/jupyter-python`
   - In the demo directory, type the following command: `pip install -r requirements.txt`

3. Run Jupyter Lab with `jupyter lab`. Click on the notebook at the left.

## Optional

### Use Voila to run the notebook in *dashboard mode*

If you want to run the notebook in dashboard mode, without showing all the
code, you can use Voila. Instead of typing `jupyter lab` type `voila
--TagRemovePreprocessor.remove_cell_tags=hide` and click on the `object_detection_demo` notebook.
In *dashboard mode* you cannot upload your own video, but video's that you already uploaded in Jupyter Lab
can be used in *dashboard mode.

### Use Public models from Open Model Zoo by installing and configuring the Model Optimizer

This demo works with models in OpenVINO IR format. Models from the Open Model
Zoo that are in the intel subdirectory are already in this format. By default,
the notebook only uses the models that are already converted. Models in the
public subdirectory need to be converted to IR format with the Model Optimizer.
This is supported in the notebook by setting `CONVERT_MODELS` to `True`. The
model optimizer needs to be installed and configured separately. Follow the
installation instructions at
https://docs.openvinotoolkit.org/2021.2/installation_guides.html, including the
steps under *Configure the Model Optimizer*.
