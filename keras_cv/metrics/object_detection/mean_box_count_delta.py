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
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv import bounding_box


class MeanBoxCountDelta(keras.metrics.Metric):
    """MeanBoxCountDelta counts the difference of counts of predicted and true boxes.

    MeanBoxCountDelta looks at the number of boxes in the ground truth dataset
    and counts the delta between that number and the number of boxes your object
    detection model predicted for that image. This is primarily useful when
    attempting to tune the confidence threshold of your
    `MultiClassNonMaxSuppression` layer in an object detection model.  If this
    metric is high, it indicates that your model is making too many or too few
    predictions.  Ideally this metric will be zero.

    Usage:

    ```python
    mean_box_count_delta = keras_cv.metrics.MeanBoxCountDelta()
    y_true = {
        "classes": [[0, -1, -1]],
        "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
    }
    y_pred = {
        "classes": [[0, 1, 1, -1]],
        "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
    }
    mean_box_count_delta.update_state(y_true, y_pred)
    print(mean_box_count_delta.result())
    # > 2.0
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.delta_sum = self.add_weight(
            name="delta_sum",
            shape=(),
            dtype=self.compute_dtype,
            initializer=initializers.Zeros(),
        )
        self.samples = self.add_weight(
            name="samples",
            shape=(),
            dtype=self.compute_dtype,
            initializer=initializers.Zeros(),
        )

    def reset_state(self):
        self.delta_sum.assign(tf.zeros_like(self.delta_sum))
        self.samples.assign(tf.zeros_like(self.samples))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Args:
            y_true: a bounding box Tensor in corners format.
            y_pred: a bounding box Tensor in corners format.
            sample_weight: Currently unsupported.
        """
        del sample_weight
        y_true = bounding_box.ensure_tensor(y_true)
        y_pred = bounding_box.ensure_tensor(y_pred)
        y_pred = bounding_box.to_dense(y_pred)
        y_true = bounding_box.to_dense(y_true)

        ground_truth_boxes = tf.cast(y_true["classes"] != -1, tf.int32)
        predicted_boxes = tf.cast(y_pred["classes"] != -1, tf.int32)

        ground_truth_boxes = tf.math.reduce_sum(ground_truth_boxes, axis=-1)
        predicted_boxes = tf.math.reduce_sum(predicted_boxes, axis=-1)

        delta_sum = tf.math.abs(ground_truth_boxes - predicted_boxes)
        self.delta_sum.assign_add(
            tf.cast(tf.math.reduce_sum(delta_sum), self.compute_dtype)
        )
        self.samples.assign_add(
            tf.cast(tf.shape(y_pred["classes"])[0], self.compute_dtype)
        )

    def result(self):
        return self.delta_sum / self.samples

    def get_config(self):
        return super().get_config()
