# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class AnchorGenerator():
  """Utility to generate anchors for a multiple feature maps.

  Example:
  ```python
  anchor_gen = AnchorGenerator([32, 64], [.5, 1., 2.],
    strides=[16, 32])
  anchors = anchor_gen([512, 512, 3])
  ```

  """

  def __init__(self,
               anchor_sizes,
               scales,
               aspect_ratios,
               strides,
               clip_boxes=False):
    """Constructs multiscale anchors.

    Args:
      anchor_sizes: A list of int represents the anchor size for each scale. The
        anchor height will be `anchor_size / sqrt(aspect_ratio)`, anchor width
        will be `anchor_size * sqrt(aspect_ratio)` for each scale.
      scales: A list/tuple, or a list/tuple of a list/tuple of positive
        floats representing the actual anchor size to the base `anchor_size`.
      aspect_ratios: A list/tuple, or a list/tuple of a list/tuple of positive
        floats representing the ratio of anchor width to anchor height.
      strides: A list/tuple of ints represent the anchor stride size between
        center of anchors at each scale.
      clip_boxes: Boolean to represents whether the anchor coordinates should be
        clipped to the image size. Defaults to `False`.
    Input shape: the size of the image, `[H, W, C]`
    Output shape: the size of anchors concat on each level, `[(H /
      strides) * (W / strides), K * 4]`
    """
    # aspect_ratio is a single list that is the same across all levels.
    aspect_ratios = maybe_map_structure_for_anchor(aspect_ratios, anchor_sizes)
    scales = maybe_map_structure_for_anchor(scales, anchor_sizes)
    if isinstance(anchor_sizes, dict):
      self.anchor_generators = {}
      for k in anchor_sizes.keys():
        self.anchor_generators[k] = _SingleAnchorGenerator(
            anchor_sizes[k], scales[k], aspect_ratios[k], strides[k],
            clip_boxes)
    elif isinstance(anchor_sizes, (list, tuple)):
      self.anchor_generators = []
      for anchor_size, scale_list, ar_list, stride in zip(
          anchor_sizes, scales, aspect_ratios, strides):
        self.anchor_generators.append(
            _SingleAnchorGenerator(anchor_size, scale_list, ar_list, stride,
                                   clip_boxes))

  def __call__(self, image_size):
    anchor_generators = tf.nest.flatten(self.anchor_generators)
    results = [anchor_gen(image_size) for anchor_gen in anchor_generators]
    return tf.nest.pack_sequence_as(self.anchor_generators, results)
