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

import tensorflow as tf

from keras_cv import bounding_box
from keras_cv import layers
from keras_cv.applications.object_detection.retina_net.__internal__ import utils


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset.
      confidence_threshold: Minimum class probability, below which detections are
        pruned.
      nms_iou_threshold: IOU threshold for the NMS operation.
      max_detections_per_class: Maximum number of detections to retain per class.
      max_detections: Maximum number of detections to retain across all classes.
      box_variance: The scaling factors used to scale the bounding box predictions.
    """

    def __init__(
        self,
        num_classes,
        bounding_box_format,
        confidence_threshold=0.05,
        iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.non_max_suppression = layers.NonMaxSuppression(
            bounding_box_format=bounding_box_format,
            num_classes=num_classes,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            max_detections_per_class=max_detections_per_class,
        )
        self._anchor_box = utils.AnchorBox(bounding_box_format=bounding_box_format)
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        # TODO correctness check
        return boxes

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        boxes = tf.concat([boxes, cls_predictions], axis=-1)

        return self.non_max_suppression(boxes)
