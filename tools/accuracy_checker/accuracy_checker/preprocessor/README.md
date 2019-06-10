# Preprocessors

Preprocessor is function which processes input data before model inference.
Every preprocessor has parameters available for configuration.
Accuracy Checker supports following set of preprocessors:

* `resize` - resizing the image to a new width and height.
  * `dst_width` and `dst_height` are destination width and height for image resizing respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
  * `resize_realization` - parameter specifies functionality of which library will be used for resize: `opencv`, `pillow` or `tf` (default `opencv` is used). For enabling `tf` you need to install Tensorflow first.
  For compatibility with previous releases you can also use boolean constants for selection resizing backend:
    * `use_pillow` parameter specifies usage of Pillow library for resizing.
    * `use_tensorflow` parameter specifies usage of TensorFlow Image for resizing. Requires TensorFlow installation.
    Accuracy Checker uses OpenCV as default image reader.
  * `interpolation` specifies method that will be used.
    Possible values depend on image processing library:
      * **OpenCV**: Nearest, Linear, Cubic, Area, Max, Lanczos4, Bits, Bits32
      * **Pillow**: None, Nearest, Cubic, Bicubic, Box, Bilinear, Lanczos, Antialias, Hamming, Linear
      * **TensorFlow**: Bilinear, Area, Bicubic
      `Linear` used as default for OpenCV, `Bilinear` as default for Pillow and TensorFlow. 
  * `aspect_ratio_scale` allows resize with changing or saving image aspect ratio. May be done using one of these ways: 
    - `width` - rescale width (height has fixed size, provided as `dst_height` or `size`, width size will be rescaled to save aspect ratio).
    - `height` - rescale height (width has fixed size, provided as `dst_width` or `size`, height size will be rescales to save aspect ratio).
    - `greater` - rescale greater from image sizes (smaller dimension has fixed size, greater will be rescaled to save aspect ratio)
    - `fit_to_window` - adaptive resize keeping aspect ratio for fit image into window with fixed size `[dst_height x dst_width]`,
         but trying to make the image as big as possible.
    - `frcnn_keep_aspect_ratio` - adaptive resize keeping aspect ratio for fit image into window with fixed size `[max_size x max_size]`,
         but trying to make the minimal dimension of image to be equal to `min_size` or as close to `min_size` as possible, where    
         `min_size = min(dst_width, dst_height)`,
         `max_size = max(dst_width, dst_height)`.
* `auto_resize` - automatic resizing image to input layer shape. (supported only for one input layer case, use OpenCV for image resize)
* `normalization` - changing the range of pixel intensity values.
  * `mean` values which will be subtracted from image channels.
     You can specify one value for all channels or list of comma separated channel-wise values.
  * `std` specifies values, on which pixels will be divided.
     You can specify one value for all channels or list of comma separated channel-wise values.

     These parameters support work with precomputed values of frequently used datasets (e.g. `cifar10` or `imagenet`).

* `bgr_to_rgb` - reversing image channels. Convert image in BGR format to RGB.
* `bgr_to_gray` - converting image in BGR to grayscale color space.
* `flip` - image mirroring around specified axis.
  * `mode` specifies the axis for flipping (`vertical` or `horizontal`).
* `crop` - central cropping for image.
  * `dst_width` and `dst_height` are destination width and height for image resizing respectively. You can also use `size` instead in case when destination sizes are equal or
  `central_fraction` to define fraction of size to crop (float value (0, 1]))
  * `use_pillow` parameter specifies usage of Pillow library for cropping.
* `crop_rectangle` - cropping region of interest using coordinates given as annotation metadata.
* `extend_around_rect` - scaling region of interest using annotation metadata.
  * `augmentation_param` is scale factor for augmentation.
* `point_aligment` - aligning keypoints stored in annotation metadata.
  * `draw_points` - allows visualize points.
  * `normalize` - allows to use normalization for keypoints.
  * `dst_width` and `dst_height` are destination width and height for keypoints resizing respectively. You can also use `size` instead in case when destination sizes are equal.
* `padding` - padding for image.
  * `stride` - stride for padding.
  * `pad_value` - value for filling space around original image.
  * `dst_width` and `dst_height` are destination width and height for padded image respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
  * `pad_type` - padding space location. Supported: `center`, `left_top`, `right_bottom` (Default is `center`).
  * `use_numpy` - allow to use numpy for padding instead default OpenCV.
* `tiling` - image tiling.
  * `margin` - margin for tiled fragment of image.
  * `dst_width` and `dst_height` are destination width and height of tiled fragment respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
* `crop3d` - central cropping for 3D data.
  * `dst_width`, `dst_height` and `dst_volume` are destination width, height and volume for cropped 3D-image respectively.
    You can also use `size` instead in case when destination sizes are equal for all three dimensions.
* `normalize3d` - normalizing 3D-images using mean and std values per channel of current image for subtraction and division respectively.
* `tf_convert_image_dtype` - cast image values to floating point values in range [0, 1]. Requires Tensorflow installation.
