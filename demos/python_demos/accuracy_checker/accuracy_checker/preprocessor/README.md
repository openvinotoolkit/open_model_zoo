# Preprocessors

Preprocessor is function which processes input data before model inference.
Every preprocessor has parameters available for configuration.
Accuracy Checker supports following set of preprocessors:

* `resize` - resizing the image to a new width and height.
  * `dst_width` and `dst_height` are destination width and height for image resizing respectively.
    You can also use `size` instead in case when destination sizes are equal for both dimensions.
  * `use_pil` parameter specifies usage of Pillow library for resizing.
    Accuracy Checker uses OpenCV as default image reader.
  * `interpolation` specifies method that will be used.
    Possible values depend on image processing library:
      * **OpenCV**: Nearest, Linear, Cubic, Area, Max, Lanczos4, Bits, Bits32
      * **Pillow**: None, Nearest, Cubic, Bicubic, Box, Bilinear, Lanczos, Antialias, Hamming
  * `rescale_greater_dim` allows save image proportions rescaling greater image dimention on destination size.

* `normalization` - changing the range of pixel intensity values.
  * `mean` values which will be subtracted from image channels.
     You can specify one value for all channels or list of comma separated channel-wise values.
  * `std` specifies values, on which pixels will be divided.
     You can specify one value for all channels or list of comma separated channel-wise values.

     These parameters support work with precomputed values of frequently used datasets (e.g. `cifar10` or `imagenet`).

* `bgr_to_rgb` - reversing image channels. Convert image in BGR format to RGB.
* `flip` - image mirroring around specified axis.
  * `mode` specifies the axis for flipping (`vertical` or `horizontal`).
* `crop` - central cropping for image.
  * `dst_width` and `dst_height` are destination width and height for image resizing respectively. You can also use `size` instead in case when destination sizes are equal.
* `crop_rectangle` - cropping region of interest using coordinates given as annotation metadata.
* `extend_around_rect` - scaling region of interest using annotation metadata.
  * `augmentation_param` is scale factor for augmentation.
* `point_aligment` - aligning keypoints stored in annotation metadata.
  * `draw_points` - allows visualize points.
  * `normalize` - allows to use normalization for keypoints.
  * `dst_width` and `dst_height` are destination width and height for keypoints resizing respectively. You can also use `size` instead in case when destination sizes are equal.
