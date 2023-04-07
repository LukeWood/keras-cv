# Copyright 2023 The KerasCV Authors
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
from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf
from tensorflow import keras


def encode_box_to_deltas(
    anchors,
    boxes,
    anchor_box_format,
    bounding_box_format,
    variance=None,
):
    """Converts bounding_boxes from `center_yxhw` to delta format."""
    if variance is not None:
        if tf.is_tensor(variance):
            var_len = variance.get_shape().as_list()[-1]
        else:
            var_len = len(variance)
        if var_len != 4:
            raise ValueError(f"`variance` must be length 4, got {variance}")
    encoded_anchors = convert_format(
        anchors,
        source=anchor_format,
        target="center_yxhw",
    )
    boxes = convert_format(
        boxes,
        source=box_format,
        target="center_yxhw",
    )
    anchor_dimensions = tf.maximum(
        encoded_anchors[..., 2:], keras.backend.epsilon()
    )
    box_dimensions = tf.maximum(boxes[..., 2:], keras.backend.epsilon())
    # anchors be unbatched, boxes can either be batched or unbatched.
    boxes_delta = tf.concat(
        [
            (boxes[..., :2] - encoded_anchors[..., :2]) / anchor_dimensions,
            tf.math.log(box_dimensions / anchor_dimensions),
        ],
        axis=-1,
    )
    if variance is not None:
        boxes_delta /= variance
    return boxes_delta


def decode_deltas_to_boxes(
    anchors,
    boxes_delta,
    anchor_box_format,
    bounding_box_format,
    variance=None,
):
    """Converts bounding_boxes from delta format to `center_yxhw`."""
    if variance is not None:
        if tf.is_tensor(variance):
            var_len = variance.get_shape().as_list()[-1]
        else:
            var_len = len(variance)
        if var_len != 4:
            raise ValueError(f"`variance` must be length 4, got {variance}")
    tf.nest.assert_same_structure(anchors, boxes_delta)

    def decode_single_level(anchor, box_delta):
        encoded_anchor = convert_format(
            anchor,
            source=anchor_format,
            target="center_yxhw",
        )
        if variance is not None:
            box_delta = box_delta * variance
        # anchors be unbatched, boxes can either be batched or unbatched.
        box = tf.concat(
            [
                box_delta[..., :2] * encoded_anchor[..., 2:]
                + encoded_anchor[..., :2],
                tf.math.exp(box_delta[..., 2:]) * encoded_anchor[..., 2:],
            ],
            axis=-1,
        )
        box = convert_format(box, source="center_yxhw", target=box_format)
        return box

    if isinstance(anchors, dict) and isinstance(boxes_delta, dict):
        boxes = {}
        for lvl, anchor in anchors.items():
            boxes[lvl] = decode_single_level(anchor, boxes_delta[lvl])
        return boxes
    else:
        return decode_single_level(anchors, boxes_delta)
