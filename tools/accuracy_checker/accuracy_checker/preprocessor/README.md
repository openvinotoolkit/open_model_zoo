# Preprocessors

Preprocessor is function which processes input data before model inference.
Every preprocessor has parameters available for configuration.
Accuracy Checker supports following set of preprocessors:

* `resize` - resizing the image to a new width and height.
  * `dst_width` and `dst_height` are destination width and height for image resizing respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
  * `resize_realization` - parameter specifies functionality of which library will be used for resize: `opencv`, `pillow` or `tf` (default `opencv` is used). For enabling `tf` you need to install TensorFlow first.
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
    - `ctpn_keep_aspect_ratio` - adaptive resize keeping aspect ratio for fit image into window with fixed size `[max_size x min_size]` using this algorithm:
      1. Try to resize min original image size to minimal destination size.
      2. If scaled max size greater than maximal destination size, rescale minimal size to get max size equal to max destination size.
    - `east_keep_aspect_ratio` - adaptive resize keeping aspect ratio using this algorithm:
      1. If max image size greater max destination size, make max image size equal to max destination size.
      2. Make image height and width divisible by min destination size without remainder.
* `auto_resize` - automatic resizing image to input layer shape. (supported only for one input layer case, use OpenCV for image resize)
* `normalization` - changing the range of pixel intensity values.
  * `mean` values which will be subtracted from image channels.
     You can specify one value for all channels or list of comma separated channel-wise values.
  * `std` specifies values, on which pixels will be divided.
     You can specify one value for all channels or list of comma separated channel-wise values.
     These parameters support work with precomputed values of frequently used datasets (e.g. `cifar10` or `imagenet`).
* `resize3d` - resizing 3d image (e.g. MRI scans) to new size:
  * `size` in format `(H,W,D)`. All values will be interpolated with 1st-order spline.
* `crop_brats`  -  performing crop of 3d images (e.g. MRI scans) by cropping all non-zero voxels. Also sets bounding boxes for `segmentation_prediction_resample` preprocessor (see [Postprocessors](../postprocessor/README.md))
* `normalize_brats` - normalization of 3d images (e.g. MRI scans) with z-score normalization
  * `masked` - specifies type of masking:
    * `none` for not applying mask
    * `ignore` for ignoring "empty" voxels in statistic calculation
    * `nullify` for nullifying initially "empty" voxels at the end
    * `all` for `ignore` and `nullify`
  * `cutoff` - cuts minimum and value to `-cutoff` and `cutoff` respectively
  * `shift_value` - adds to all values
  * `normalize_value` - divides all values
* `swap_modalities` - swapping modalities of MRI scan (works as channel swapping)
  * `modality_order` - new order
* `bgr_to_rgb` - reversing image channels. Convert image in BGR format to RGB.
* `bgr_to_gray` - converting image in BGR to gray scale color space.
* `rgb_to_bgr` - reversing image channels. Convert image in RGB format to BGR.
* `rgb_to_gray` - converting image in RGB to gray scale color space.
* `bgr_to_yuv` - converting image in BGR to YUV.
  * `split_channels` - split image channels to independent input data after conversion (Optional, default `False`).
* `rgb_to_yuv` - converting image in RGB to YUV.
  * `split_channels` - split image channels to independent input data after conversion (Optional, default `False`).
* `select_channel` - select channel only one specified channel from multichannel image.
  * `channel` - channel id in image (e.g. if you read image in RGB and want to select green channel, you need to specify 1 as channel)
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
  * `numpy_pad_mode` - if using numpy for padding, numpy padding mode, including constant, edge, mean, etc. (Default is `constant`)
* `tiling` - image tiling.
  * `margin` - margin for tiled fragment of image.
  * `dst_width` and `dst_height` are destination width and height of tiled fragment respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
* `crop3d` - central cropping for 3D data.
  * `dst_width`, `dst_height` and `dst_volume` are destination width, height and volume for cropped 3D-image respectively.
    You can also use `size` instead in case when destination sizes are equal for all three dimensions.
* `normalize3d` - normalizing 3D-images using mean and std values per channel of current image for subtraction and division respectively.
* `tf_convert_image_dtype` - cast image values to floating point values in range [0, 1]. Requires TensorFlow installation.
* `decode_by_vocabulary` - Decode words to set of indexes using model vocab.
  * `vocabulary_file` - path to vocabulary file for decoding. Path can be prefixed with `--models` argument.
  * `unk_index` - index of unknown symbol in vocab.
*  `pad_with_eos` - supplement the input sequence to a specific size using a line terminator character or index.
  * `eos_symbol` or `eos_index` - line terminator symbol or index of this symbol in vocab for encoded sequence respectively.
  *  `sequence_len` - length of sequence after supplement.
* `centernet_affine_transform` - CenterNet affine transformation, used for image resizing.
  * `dst_width` and `dst_height` are destination width, and height for image. You can also use size instead in case when destination sizes are equal.
  * `scale` - scale factor for image (default is 1).
* `free_form_mask` - Applies free-form mask to the image.
  * `parts` - Number of parts to draw mask.
  * `max_brush_width` - Maximum brush width to draw mask.
  * `max_length` - Maximum line length to draw mask.
  * `max_vertex` - Maximum number vertex to draw mask.
* `rect_mask` - Applies rectangle mask to the image.
  * `dst_width` and `dst_height` are width, and height of mask. You can also use `size` instead in case when destination sizes are equal.
* `custom_mask` - Applies masks from custom mask dataset.
  * `mask_dir` - path to mask dataset to be used for inpainting.
  * `inverse_mask` - inverse mask before apply
  * `mask_loader` - which reader to use to load masks. The following readers can be used:
    * `opencv_imread` - read images using OpenCV library. Default color space is BGR.
    * `pillow_imread` - read images using Pillow library. Default color space is RGB.
    * `scipy_imread` - read images using similar approach as in `scipy.misc.imread`.
    * `numpy_reader` - read numpy dumped files.
    * `tf_imread`- read images using TensorFlow. Default color space is RGB. Requires TensorFlow installation.
* `warp_affine` - warp affine transformation. (supported only with OpenCV)
  * `src_landmarks` - source landmarks to set as markers for the warp affine transformation.
  * `dst_landmarks` - destination and target landmarks to transform `src_landmarks` to.
* `resample_audio` - converts audio to new sample rate
  * `sample_rate` - sets new sample rate
* `clip_audio` - slices audio into several parts with equal duration
  * `duration` - sets duration of each clip in seconds or samples (use `samples` suffix), e.g. `1.5`, `16000samples`
  * `overlap` - sets overlapping for clips in percents or samples (use `%` or `samples` suffixes respectively) (no overlapping by default), e.g. `25%`, `4000samples`
  * `max_clips` - sets the maximum number of clips (clips all record by default)
* `audio_normalization` - normalize audio record with mean sample subtraction and division on standard deviation of samples.
* `similarity_transform_box` - apply to image similarity transformation to get rectangle region stored in annotation metadata/
    * `box_scale` - box scale factor (Optional, default 1).
    * `dst_width` and `dst_height` are destination width and height for transformed image respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
* `face_detection_image_pyramid` - image pyramid for face detection
  * `min_face_ratio` - minimum face ratio to image size.
  * `resize_scale` - scale factor for pyramid layers.
* `face_patch` - crops faces detected in previous stage model from input image with vertical and horizontal scaling.
  * `scale_width` - value to scale width relative to the original candidate width.
  * `scale_height` - value to scale height relative to the original candidate height.
