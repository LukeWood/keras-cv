import tensorflow as tf
from tensorflow import keras

from keras_cv import bounding_box


class AnchorGenerator(keras.layers.Layer):
    """AnchorGenerator generates anchors for multiple feature maps.

    AnchorGenerator takes multiple scales and generates anchor boxes based on the anchor
    sizes, scales, aspect ratios, and strides provided.  To invoke AnchorGenerator, call
    it on the image that needs anchor boxes.

    Args:
      bounding_box_format: The format of bounding boxes to generate. Refer
        [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
        for more details on supported bounding box formats.
      scales: A list or dictionary where each entry is a float corresponding to a
        multiplier that is multiplied by each `anchor_size` to define a level in the.
        Feature pyramid.  If `scales` is a list, the output dictionary of the
        `AnchorGenerator` will contain keys such as "level_1", "level_2", and so on.
      sizes: A list of integers that represent the length of a single side of an
        anchor box for each level, or a dictionary of integer lists with each key
        representing the sizes to use at each level. For each size, anchor height will
        be `anchor_size / sqrt(aspect_ratio)`, and anchor width will be
        `anchor_size * sqrt(aspect_ratio)`.  This is repeated for each scale and aspect
        ratio.
      aspect_ratios: A list of floats representing the ratio of anchor width to height,
        or a dictionary of lists of floats where each key represents the aspect ratios
        to use a each level.
      strides: list of ints that represent the anchor stride size between
        center of anchors at each scale, or dictionary where each entry is a list
        matching the aforementioned criteria for each level.  The length of strides must
        be equal to the length of sizes, or if a dictionary is provided each entry must
        match.
      clip_boxes: Whether or not to clip generated anchor boxes to the image size.
        Defaults to `False`.

    Usage:
    ```python
    # TODO(lukewood): fill out a construction.
    image = tf.random.uniform((512, 512, 3))
    anchors = anchor_gen(image)
    ```

    Input shape: an image with shape `[H, W, C]`
    Output: a dictionary with integer keys corresponding to each level of the feature
        pyramid.  The size of the anchors at each level will be
        `(H/strides * W/strides, 4)`.
    """

    def __init__(
        self,
        bounding_box_format,
        sizes,
        scales,
        aspect_ratios,
        strides,
        clip_boxes=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        # aspect_ratio is a single list that is the same across all levels.

        scales = AnchorGenerator._pack_parameter_to_dict(scales, "scales")
        sizes = AnchorGenerator._broadcast_parameter_to_scales(sizes, "sizes", scales)
        strides = AnchorGenerator._broadcast_parameter_to_scales(
            strides, "strides", scales
        )
        AnchorGenerator._validate_sizes_and_strides(sizes, strides)
        aspect_ratios = AnchorGenerator._broadcast_parameter_to_scales(
            aspect_ratios, "aspect_ratios", scales
        )

        self.anchor_generators = {}
        for level in scales.keys():
            self.anchor_generators[level] = _SingleAnchorGenerator(
                scales[level],
                sizes[level],
                aspect_ratios[level],
                strides[level],
                clip_boxes,
            )

    @staticmethod
    def _validate_sizes_and_strides(sizes, strides):
        for level in sizes:
            sizes_at_level = sizes[level]
            strides_at_level = strides[level]
            if len(sizes_at_level) != len(strides_at_level):
                raise ValueError(
                    "Expected `sizes` and `strides` to be of the same length. At level "
                    f"'{level}' sizes={sizes_at_level}, strides={strides_at_level}. "
                    "Please ensure that `sizes` and `strides` match all levels."
                )

    @staticmethod
    def _format_sizes_and_strides(sizes, strides):
        result_sizes = AnchorGenerator._ensure_param_is_levels_dict(sizes, "sizes")
        result_strides = AnchorGenerator._ensure_param_is_levels_dict(
            strides, "strides"
        )

        if sorted(result_strides.keys()) != sorted(result_sizes.keys()):
            raise ValueError(
                "Expected sizes and strides to be either lists of"
                "the same length, or dictionaries with the same keys.  Received "
                f"sizes={sizes}, strides={strides}"
            )

        return result_sizes, result_strides

    @staticmethod
    def _broadcast_parameter_to_scales(param, param_name, scales):
        if not isinstance(param, (list, dict)):
            raise ValueError(
                f"Expected `{param_name}` to be a list, or a dictionary mapping from "
                f"level names to lists of values.  Received `{param_name}={param}`."
            )

        if isinstance(param, list):
            result = {}
            for key in scales:
                result[key] = param
            param = result

        # only possible when param is a custom dict.
        if sorted(scales.keys()) != sorted(param.keys()):
            raise ValueError(
                f"Expected `{param_name}` to contain the same keys as `scales`. "
                f"Received `scales.keys()={scales.keys()}`, "
                f"`{param_name}.keys()={param.keys()}`."
            )
        return param

    @staticmethod
    def _pack_parameter_to_dict(param, param_name):
        """Takes a param and its name, converts lists to dictionaries of levels.
        For example, the list [1, 2] is converted to {'level_1': 1, 'level_2': 2}.

        Raises:
            ValueError: when param is not a dict, list or tuple.
        """
        if not isinstance(param, (dict, list, tuple)):
            raise ValueError(
                f"Expected {param_name} to be a dict, list or tuple, received "
                f"{param_name}={sizes}"
            )

        if isinstance(param, dict):
            return param

        result = {}
        for i in range(len(param)):
            result[f"level_{i}"] = param[i]
        return result

    @staticmethod
    def _match_param_structure_to_sizes(params, sizes):
        """broadcast the params to match sizes."""
        # if isinstance(sizes, (tuple, list)):
        #     return [params] * len(sizes)
        if not isinstance(sizes, dict):
            raise ValueError(
                "the structure of `sizes` must be a dict, " f"received sizes={sizes}"
            )

        return tf.nest.map_structure(lambda _: params, sizes)

    def __call__(self, image):
        image_shape = tf.shape(image)
        anchor_generators = tf.nest.flatten(self.anchor_generators)
        results = [anchor_gen(image_shape) for anchor_gen in anchor_generators]
        results = tf.nest.pack_sequence_as(self.anchor_generators, results)
        for key in results:
            results[key] = bounding_box.convert_format(
                results[key],
                source="yxyx",
                target=self.bounding_box_format,
                images=image,
            )
        return results


# TODO(tanzheny): consider having customized anchor offset.
class _SingleAnchorGenerator:
    """Internal utility to generate anchors for a single feature map in `yxyx` format.

    Example:
    ```python
    anchor_gen = _SingleAnchorGenerator(32, [.5, 1., 2.], stride=16)
    anchors = anchor_gen([512, 512, 3])
    ```
    """

    def __init__(
        self,
        scale,
        sizes,
        aspect_ratios,
        stride,
        clip_boxes=False,
    ):
        """Constructs single scale anchor.

        Args:
          scale: A list/tuple, or a list/tuple of a list/tuple of positive
            floats representing the actual anchor size to the base `anchor_size`.
          sizes: A single int represents the base anchor size. The anchor
            height will be `anchor_size / sqrt(aspect_ratio)`, anchor width will be
            `anchor_size * sqrt(aspect_ratio)`.
          aspect_ratios: a list/tuple of positive floats representing the ratio of
            anchor width to anchor height.
          stride: A single int represents the anchor stride size between center of
            each anchor.
          clip_boxes: Boolean to represent whether the anchor coordinates should be
            clipped to the image size. Defaults to `False`.

        Input shape: the size of the image, `[H, W, C]`
        Output shape: the size of anchors, `[(H / stride) * (W / stride), 4]`
        """
        self.sizes = sizes
        self.scale = scale
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        self.clip_boxes = clip_boxes

    def __call__(self, image_size):
        image_height = tf.cast(image_size[0], tf.float32)
        image_width = tf.cast(image_size[1], tf.float32)

        aspect_ratios = tf.cast(self.aspect_ratios, tf.float32)
        aspect_ratios_sqrt = tf.cast(tf.sqrt(aspect_ratios), dtype=tf.float32)

        anchor_sizes = tf.cast(self.sizes, tf.float32)
        anchor_sizes_t = anchor_sizes * self.scale
        anchor_heights = anchor_sizes_t / aspect_ratios_sqrt
        anchor_widths = anchor_sizes_t * aspect_ratios_sqrt

        # anchor_heights = tf.concat(anchor_heights, axis=0)
        # anchor_widths = tf.concat(anchor_widths, axis=0)
        half_anchor_heights = tf.reshape(0.5 * anchor_heights, [1, 1, -1])
        half_anchor_widths = tf.reshape(0.5 * anchor_widths, [1, 1, -1])

        # strides can only be done on a per-item basis, so this is tricky.
        stride = tf.cast(self.stride, tf.float32)
        # [W]
        cx = tf.range(0.5 * stride, image_width, stride)
        # [H]
        cy = tf.range(0.5 * stride, image_height, stride)
        # [H, W]
        cx_grid, cy_grid = tf.meshgrid(cx, cy)
        # [H, W, 1]
        cx_grid = tf.expand_dims(cx_grid, axis=-1)
        cy_grid = tf.expand_dims(cy_grid, axis=-1)

        y_min = tf.reshape(cy_grid - half_anchor_heights, (-1,))
        y_max = tf.reshape(cy_grid + half_anchor_heights, (-1,))
        x_min = tf.reshape(cx_grid - half_anchor_widths, (-1,))
        x_max = tf.reshape(cx_grid + half_anchor_widths, (-1,))

        # [H * W * K, 1]
        y_min = tf.expand_dims(y_min, axis=-1)
        y_max = tf.expand_dims(y_max, axis=-1)
        x_min = tf.expand_dims(x_min, axis=-1)
        x_max = tf.expand_dims(x_max, axis=-1)

        if self.clip_boxes:
            y_min = tf.maximum(tf.minimum(y_min, image_height), 0.0)
            y_max = tf.maximum(tf.minimum(y_max, image_height), 0.0)
            x_min = tf.maximum(tf.minimum(x_min, image_width), 0.0)
            x_max = tf.maximum(tf.minimum(x_max, image_width), 0.0)

        # [H * W * K, 4]
        return tf.concat([y_min, x_min, y_max, x_max], axis=-1)
