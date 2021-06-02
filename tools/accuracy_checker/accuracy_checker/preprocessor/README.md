# Preprocessors

Preprocessor is a class which processes input data before model inference. Every preprocessor has parameters available for configuration. The preprocessor and its parameters are set through the configuration file. Preprocessors are provided in `datasets` section of configuration file to use specific preprocessor.

## Supported Preprocessors

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
    - `min_ratio` - rescale width and height according to minimal ratio `source_size / destination_size`.
    - `mask_rcnn_benchmark_aspect_ratio` - rescale image size according [preprocessing](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/mask-rcnn/README.md#preprocessing-steps) for maskrcnn-benchmark models in ONNX zoo
    - `ppcrnn_ratio` - calculate scales in the following way:
      1. find original image ratio (input_width / input_height)
      2. if `dst_height` * ratio larger then `dst_width`, then `dst_width` = 32 * ratio
      3. Otherwise `dst_width` = `dst_height` * ratio
* `factor` -  destination size for aspect ratio resize must be divisible by a given number without remainder.
  Please pay attention that this parameter only works with `aspect_ratio_scale` parameters.
* `auto_resize` - automatic resizing image to input layer shape. (supported only for one input layer case, use OpenCV for image resize)
* `normalization` - changing the range of pixel intensity values.
  * `mean` values which will be subtracted from image channels.
     You can specify one value for all channels or list of comma separated channel-wise values.
  * `std` specifies values, on which pixels will be divided.
     You can specify one value for all channels or list of comma separated channel-wise values.
     These parameters support work with precomputed values of frequently used datasets (e.g. `cifar10` or `imagenet`).
  * `images_only` - prevent usage normalization for non-image inputs in multi input mode (Optional, default `False`).
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
  * `shrink_uv` - resize uv-channels in 1:2 resolution to y-channel (Optional, available only with `split_channels` combination).
* `rgb_to_yuv` - converting image in RGB to YUV.
  * `split_channels` - split image channels to independent input data after conversion (Optional, default `False`).
  * `shrink_uv` - resize uv-channels in 1:2 resolution to y-channel (Optional, available only with `split_channels` combination).
* `bgr_to_nv12` - converting BGR image to NV12 format.
* `rgb_to_nv12` - converting RGB image to NV12 format.
* `nv12_to_bgr` - converting NV12 data to BGR format.
* `nv12_to_rgb` - converting NV12 data to RGB format.
* `bgr_to_ycrcb` - converting image in BGR to YCrCb.
  * `split_channels` - split image channels to independent input data after conversion (Optional, default `False`).
* `rgb_to_ycrcb` - converting image in RGB to YCrCb.
  * `split_channels` - split image channels to independent input data after conversion (Optional, default `False`).
* `bgr_to_lab` - converts image in RGB format to LAB.
* `rgb_to_lab` - converts image in BGR format to LAB.
* `select_channel` - select channel only one specified channel from multichannel image.
  * `channel` - channel id in image (e.g. if you read image in RGB and want to select green channel, you need to specify 1 as channel)
* `flip` - image mirroring around specified axis.
  * `mode` specifies the axis for flipping (`vertical` or `horizontal`).
  * `merge_with_original` - allows addition flipped image to original (Optional, default `False`, original image will be replaced with flipped).
* `crop` - central cropping for image.
  * `dst_width` and `dst_height` are destination width and height for image resizing respectively. You can also use `size` instead in case when destination sizes are equal,
  `central_fraction` to define fraction of size to crop (float value (0, 1])) or `max_square` for cropping central part for image by minimal image size (`True` value for enabling this feature).
  * `use_pillow` parameter specifies usage of Pillow library for cropping.
* `crop_rectangle` - cropping region of interest using coordinates given as annotation metadata.
* `extend_around_rect` - scaling region of interest using annotation metadata.
  * `augmentation_param` is scale factor for augmentation.
* `point_alignment` - aligning keypoints stored in annotation metadata.
  * `draw_points` - allows visualize points.
  * `normalize` - allows to use normalization for keypoints.
  * `dst_width` and `dst_height` are destination width and height for keypoints resizing respectively. You can also use `size` instead in case when destination sizes are equal.
* `corner_crop` - Corner crop of the image.
  * `dst_width` and `dst_height` are destination width and height
  * `corner_type` is type of the corner crop. Options are:
    * `top-left`
    * `top-right`
    * `bottom-left`
    * `bottom-right`
  Default choice is `top-left`
* `crop_or_pad` - performs central cropping if original image size greater then destination size and padding in case, when source size lower than destination. Padding filling value is 0, realization - right-bottom.
  * `dst_width` and `dst_height` are destination width and height for keypoints resizing respectively. You can also use `size` instead in case when destination sizes are equal.
* `crop_image_with_padding`- crops to center of image with padding then scales image size.
  * `size` - destination image height/width dimension,
  * `crop_padding` - the padding size to use when centering the crop.
* `padding` - padding for image.
  * `stride` - stride for padding.
  * `pad_value` - value for filling space around original image.
  * `dst_width` and `dst_height` are destination width and height for padded image respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
  * `pad_type` - padding space location. Supported: `center`, `left_top`, `right_bottom` (Default is `center`).
  * `use_numpy` - allow to use numpy for padding instead default OpenCV.
  * `numpy_pad_mode` - if using numpy for padding, numpy padding mode, including constant, edge, mean, etc. (Default is `constant`).
  * `enable_resize` - allow resize image to destination size, if source image greater that destination (Optional, default `False`).
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
* `decode_by_sentence_piece_bpe_tokenizer` - Decode words to set of indexes using SentencePieceBPETokenizer.
  * `vocabulary_file` - path to vocabulary file for decoding. Path can be prefixed with `--models` argument.
  * `merges_file` - path to merges file for decoding. Path can be prefixed with `--models` argument.
  * `sos_symbol` - string representation of start_of_sentence symbol (default=`<s>`).
  * `eos_symbol` - string representation of end_of_sentence symbol (default=`</s>`).
  * `add_symbols` - add sos/eos symbols to sentence (default=True).
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
  * `inverse_mask` - Allows mask inversion (1 - real image, 0 - masked area). Optional, default `False` (0 - real image, 1- masked area).
* `rect_mask` - Applies rectangle mask to the image.
  * `dst_width` and `dst_height` are width, and height of mask. You can also use `size` instead in case when destination sizes are equal.
* `inverse_mask` - Allows mask inversion (1 - real image, 0 - masked area). Optional, default `False` (0 - real image, 1- masked areaa).
* `custom_mask` - Applies masks from custom mask dataset.
  * `mask_dir` - path to mask dataset to be used for inpainting.
  * `inverse_mask` - inverse mask before apply
  * `mask_loader` - which reader to use to load masks. The following readers can be used:
    * `opencv_imread` - read images using OpenCV library. Default color space is BGR.
    * `pillow_imread` - read images using Pillow library. Default color space is RGB.
    * `scipy_imread` - read images using similar approach as in `scipy.misc.imread`.
    * `numpy_reader` - read numpy dumped files.
    * `tf_imread`- read images using TensorFlow. Default color space is RGB. Requires TensorFlow installation.
    * `inverse_mask` - Allows mask inversion (1 - real image, 0 - masked area). Optional, default `False` (0 - real image, 1- masked areaa).
* `warp_affine` - warp affine transformation. (supported only with OpenCV)
  * `src_landmarks` - source landmarks to set as markers for the warp affine transformation.
  * `dst_landmarks` - destination and target landmarks to transform `src_landmarks` to.
  * `dst_height` - destination height size.
  * `dst_width` - destination width size.
* `resample_audio` - converts audio to new sample rate
  * `sample_rate` - sets new sample rate
* `clip_audio` - slices audio into several parts with equal duration
  * `duration` - sets duration of each clip in seconds or samples (use `samples` suffix), e.g. `1.5`, `16000samples`
  * `overlap` - sets overlapping for clips in percents or samples (use `%` or `samples` suffixes respectively) (no overlapping by default), e.g. `25%`, `4000samples`
  * `max_clips` - sets the maximum number of clips (clips all record by default)
* `audio_normalization` - normalize audio record with mean sample subtraction and division on standard deviation of samples.
* `audio_to_mel_spectrogram` - performs all needed preprocessing to calculate MEL spectrogram from time-domain audio signal
  * `window_size` - size of time-domain signal frame, seconds
  * `window_stride` - intersection of frames in time-domain, seconds
  * `window`- weighting window type, possible choices:
    * `hann` - applies Hanning window to each signal frame
    * `hamming` - applies Hamming window to each signal frame
    * `blackman` - applies Blackman window to each signal frame
    * `bartlett` - applies Bartlett window to each signal frame
    * `none` - no window
  * `n_fft` - STFT base, samples
  * `n_filt` - number of MEL filters
  * `splicing` - number of sequentially concastenated MEL spectrums
  * `sample_rate` - audio sampling frequency, Hz
  * `pad_to` - desired length of features
  * `preemph` - preemph factor
  * `log` - applies log() to MEL features values
  * `use_deterministic_dithering` - Controls  dithering mode:
    * `True` - there are no dithering in time-domain, fixed value from `dither` parameter added to signal spectrum
    * `False` - dithering in time-domain, random values with  `dither` magnitude added to signal spectrum
  * `dither` - dithering value
* `audio_patches` - split audio signal on patches with specified `size` for multi inference processing. If input signal can not be divided by size without remainder, signal will be padded by zeros left side.
  * `size` - patch size.
* `context_window` - add context window padding to input signal.
  * `cw_l` - left side context window padding.
  * `cw_r` - right side context window padding.
* `similarity_transform_box` - apply to image similarity transformation to get rectangle region stored in annotation metadata/
    * `box_scale` - box scale factor (Optional, default 1).
    * `dst_width` and `dst_height` are destination width and height for transformed image respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
* `face_detection_image_pyramid` - image pyramid for face detection
  * `min_face_ratio` - minimum face ratio to image size.
  * `resize_scale` - scale factor for pyramid layers.
* `candidate_crop` - crops candidates detected in previous stage model from input image with vertical and horizontal scaling.
  * `scale_width` - value to scale width relative to the original candidate width.
  * `scale_height` - value to scale height relative to the original candidate height.
* `object_crop_with_scale` - crop region from image using `center` coordinate and `scale` from annotation.
  * `dst_width` and `dst_height` are destination width and height for image cropping respectively. You can also use `size` instead in case when destination sizes are equal.
* `one_hot_encoding` - create label map based on array of indexes (analog scatter).
  * `value` - number for encoding label.
  * `base` - number for encoding other classes.
  * `axis` - axis responsible for classes.
  * `number_of_classes` - number of used classes.
* `pack_raw_image` - pack raw image to [H, W, 4] normalized image format with black level removal.
  * `black_level` - black level on the input image.
  * `ratio` - exposure scale ratio, optional, can be replaced by value from annotation if not provided.
  * `9-channels` - for packing 9 channels images (Optional, default `False`).
* `alpha` - extracts alpha-channel data from the image.
  * `channel` - number of channel to extract (Optional, default 3).
* `trimap` - concatenates image data with alpha-channel based information for cut, keep and calculation zones in image.
  * `cut_treshold` - maximum level of alpha values in cut zone. Optional, default is 0.1.
  * `keep_treshold` - minimum level of alpha values in keep zone. Optional, default is 0.9. Pixels with alpha-channel values between `cut_threshold` and `keep_treshold` are in calculation zone.

## Optimized preprocessing via OpenVINO Inference Engine
OpenVINO™ is able perform preprocessing during model execution. For enabling this behaviour you can use command line parameter `--ie_preprocessing True`.
When this option turn on, specified in config preprocessing will be translated to Inference Engine PreProcessInfo API.
**Note**: This option is available only for `dlsdk` launcher and not all preprocessing operations can be ported to Inference Engine.
Supported preprocessing:
* Resizing: `resize` without aspect_ratio_scale and with `BILINEAR` or `AREA` interpolation. Destination size is model input shape. (`auto_resize` is also can be used for resize with bilinear interpolation.)
* Color conversion: `bgr_to_rgb`, `rgb_to_bgr`, `nv12_to_bgr`, `nv12_to_rgb`
* Normalization:  `normalization` with per channel `mean` and `std`
