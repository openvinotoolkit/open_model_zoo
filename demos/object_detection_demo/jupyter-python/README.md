# Object Detection Async Demo

This demo showcases Object Detection on Open Model Zoo models with Async API.
Async API usage can improve the overall frame-rate of the application, because
rather than wait for inference to complete, the app can continue doing things
on the host, while accelerator is busy.

This notebook allows you to select a model and an input video, as well as vary
the number of streams, threads and requests for the inference.

Note: the notebook allows you to upload your own video. It is recommended to
use a short video. If you use a video that is longer than a few minutes, you
can adjust the `JUMP_FRAMES` setting to a larger value to increase inference
speed. With the default setting every tenth frame is analyzed.

Other demo objectives are:

* Video as input support via OpenCV\*
* Visualization of the resulting bounding boxes
* Comparison of different Open Model Zoo models

See the [Python Object Detection Async Demo](../python/) for more details about
the Async API, and the [Optimization
Guide](https://docs.openvinotoolkit.org/latest/_docs_optimization_guide_dldt_optimization_guide.html)
for more information on optimizing models.


## Run this demo on your computer

1. Clone the Open Model Zoo repository to your computer with `git`, or download the repository by clicking on this link[this link](https://github.com/openvinotoolkit/open_model_zoo/archive/master.zip) and unzip it.

2. Install Python 

    This notebook has been tested on Linux (Ubuntu 18.04) and Windows, with Python 3.7.9. If you have not installed Python yet, download and install it from https://www.python.org/downloads/release/python-379/. Choose a 64-bit version (not embedded). If you are on Windows, it is recommended to select the `Add Python to your PATH` option during installation.

3. Install the required Python packages
   - Open a terminal and go to the demo directory (the directory that contains this README.)
     - On Linux or MacOS, open Terminal. On Windows, open a Command Prompt (type `cmd` in the search Window in the task bar)
     - Use the `cd` command to go to the correct directory (tip: pressing the Tab key auto-expands the filenames, so you can type `cd Dow<tab>`)
   - In the demo directory, type the following command: `pip install -r requirements.txt`

4. Add the OpenVINO directory to the PATH (Windows) or LD_LIBRARY_PATH (Linux/MacOS) variable.
    - NOTE: the PATH is remembered as long as you do not close the terminal application. If you open a new terminal and want to run the notebooks again, you have to set the PATH again. 
    - On Windows, on a fresh install:
      - Find the library path by copy/pasting: `python -c "import os, sys; print((os.path.dirname(sys.executable))+'\Library\\bin')"`. If you followed the default installation, the path will be `C:\Users\USERNAME\AppData\Local\Programs\Python\Python37\Library\bin`
      - Copy the library path (with CTRL-C) and type `set PATH=` followed by CTRL-V to paste the path you just copied, followed by `;%PATH%;`
    - For other situations, follow the instructions on https://pypi.org/project/openvino-python/
 
5. Run Jupyter Lab with `jupyter lab`. Click on the notebook at the left.

## Optional

### Use Voila to run the notebook in `dashboard mode`

If you want to run the notebook in dashboard mode, without showing all the
code, you can use Voila. Instead of typing `jupyter lab` type `voila
--TagRemovePreprocessor.remove_cell_tags=hide
--TagRemovePreprocessor.remove_cell_tags=hide_temp object_detection_demo.ipynb`

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


