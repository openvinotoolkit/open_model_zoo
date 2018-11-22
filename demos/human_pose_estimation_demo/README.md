# Human Pose Estimation Demo

This demo showcases the work of multi-person 2D pose estimation algorithm. The task is to predict a pose: body skeleton, which consists of keypoints and connections between them, for every person in an input video. The pose may contain up to 18 keypoints: *ears, eyes, nose, neck, shoulders, elbows, wrists, hips, knees*, and *ankles*. Some of potential use cases of the algorithm are action recognition and behavior understanding. The following pre-trained model is delivered with the product:

* `human-pose-estimation-0001`, which is a human pose estimation network, that produces two feature vectors. The algorithm uses these feature vectors to predict human poses.

The input frame height is scaled to model height, frame width is scaled to preserve initial aspect ratio and padded to multiple of 8.

Other demo objectives are:
* Video/Camera as inputs, via OpenCV*
* Visualization of all estimated poses

## How It Works

On the start-up, the application reads command line parameters and loads human pose estimation model. Upon getting a frame from the OpenCV VideoCapture, the application executes human pose estimation algorithm and displays the results.

## Running

Running the application with the `-h` option yields the following usage message:
```sh
./human_pose_estimation_demo -h
InferenceEngine:
    API version ............ <version>
    Build .................. <number>

human_pose_estimation_demo [OPTION]
Options:

    -h                         Print a usage message.
    -i "<path>"                Required. Path to a video. Default value is "cam" to work with camera.
    -m "<path>"                Required. Path to the Human Pose Estimation model (.xml) file.
    -d "<device>"              Optional. Specify the target device for Human Pose Estimation (CPU, GPU, FPGA or MYRIAD is acceptable). Default value is "CPU".
    -pc                        Optional. Enable per-layer performance report.
    -no_show                   Optional. Do not show processed video.
    -r                         Optional. Output inference results as raw values.

```

Running the application with an empty list of options yields an error message.

To run the demo, use the pre-trained and optimized `human-pose-estimation-0001` model delivered with the product. The model is located at `<INSTALL_DIR>/deployment_tools/intel_models/`.

For example, to do inference on a CPU, run the following command:

```sh
./human_pose_estimation_demo -i <path_to_video>/input_video.mp4 -m <path_to_model>/human-pose-estimation-0001.xml -d CPU
```

## Demo Output

The demo uses OpenCV to display the resulting frame with estimated poses and text report of **FPS** - frames per second performance for the human pose estimation demo.

## See Also
* [Using Inference Engine Samples](./docs/Inference_Engine_Developer_Guide/Samples_Overview.md)
