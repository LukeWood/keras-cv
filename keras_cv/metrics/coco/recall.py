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
import warnings

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv import bounding_box
from keras_cv.bounding_box import iou as iou_lib
from keras_cv.metrics.coco import utils


class _BoxRecall(keras.metrics.Metric):
    """_BoxRecall computes recall based on varying true positive IoU thresholds.

    _BoxRecall is analagous to traditional Recall.  The primary distinction is
    that when operating in the problem domain of object detection there exists
    ambiguity in what is considered a true positive.  The _BoxRecall metric
    works by using the Intersection over Union (IoU) metric to determine whether
    or not a detection is a true positive or a false positive.  An average
    across many IoU thresholds may also be taken by passing a list to the
    `iou_thresholds` argument.

    Args:
        class_ids: The class IDs to evaluate the metric for.  To evaluate for
            all classes in over a set of sequentially labelled classes, pass
            `range(classes)`.
        bounding_box_format: Format of the incoming bounding boxes.  Supported values
            are "xywh", "center_xywh", "xyxy".
        iou_thresholds: IoU thresholds over which to evaluate the recall.  Must
            be a tuple of floats, defaults to [0.5:0.05:0.95].
        area_range: area range to constrict the considered bounding boxes in
            metric computation. Defaults to `None`, which makes the metric
            count all bounding boxes.  Must be a tuple of floats.  The first
            number in the tuple represents a lower bound for areas, while the
            second value represents an upper bound.  For example, when
            `(0, 32**2)` is passed to the metric, recall is only evaluated for
            objects with areas less than `32*32`.  If `(32**2, 1000000**2)` is
            passed the metric will only be evaluated for boxes with areas larger
            than `32**2`, and smaller than `1000000**2`.
        max_detections: number of maximum detections a model is allowed to make.
            Must be an integer, defaults to `100`.

    Usage:

    _BoxRecall accepts two dictionaries that comply with KerasCV's bounding box
    specification as inputs to it's `update_state` method.
    These dictionaries represent bounding boxes in the specified
    `bounding_box_format`.

    ```python
    coco_recall = keras_cv.metrics._BoxRecall(
        bounding_box_format='xyxy',
        max_detections=100,
        class_ids=[1]
    )
    od_model.compile(metrics=[coco_recall])
    od_model.fit(my_dataset)
    ```
    """

    def __init__(
        self,
        class_ids,
        bounding_box_format,
        iou_thresholds=None,
        area_range=None,
        max_detections=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Initialize parameter values
        self.bounding_box_format = bounding_box_format
        iou_thresholds = iou_thresholds or [
            x / 100.0 for x in range(50, 100, 5)
        ]

        self.iou_thresholds = iou_thresholds
        self.class_ids = list(class_ids)
        self.area_range = area_range
        self.max_detections = max_detections

        # Initialize result counters
        num_thresholds = len(iou_thresholds)
        num_categories = len(class_ids)

        if any([c < 0 for c in class_ids]):
            raise ValueError(
                "class_ids must be positive.  Got " f"class_ids={class_ids}"
            )

        self.true_positives = self.add_weight(
            name="true_positives",
            shape=(num_thresholds, num_categories),
            dtype=tf.int32,
            initializer=initializers.Zeros(),
        )
        self.ground_truth_boxes = self.add_weight(
            name="ground_truth_boxes",
            shape=(num_categories,),
            dtype=tf.int32,
            initializer=initializers.Zeros(),
        )

    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.ground_truth_boxes.assign(tf.zeros_like(self.ground_truth_boxes))

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not yet supported in keras_cv COCO metrics."
            )
        y_true = bounding_box.ensure_tensor(y_true, dtype=self.compute_dtype)
        y_pred = bounding_box.ensure_tensor(y_pred, dtype=self.compute_dtype)
        bounding_box.validate_format(
            y_true,
            variable_name="y_true",
        )
        bounding_box.validate_format(
            y_pred,
            variable_name="y_pred",
        )

        if y_true["boxes"].shape.rank != 3 or y_pred["boxes"].shape.rank != 3:
            raise ValueError(
                "Expected `y_true` and `y_pred` to be batched. "
                "Received "
                f"`y_true['boxes'].shape.rank={y_true['boxes'].shape.ranks}` "
                "and "
                f"`y_pred['boxes'].shape.rank={y_true['boxes'].shape.ranks}`. "
                "Expected both to be 3."
            )

        y_true = bounding_box.to_dense(y_true)
        y_pred = bounding_box.to_dense(y_pred)

        y_true = bounding_box.convert_format(
            y_true,
            source=self.bounding_box_format,
            target="xyxy",
            dtype=self.compute_dtype,
        )
        y_pred = bounding_box.convert_format(
            y_pred,
            source=self.bounding_box_format,
            target="xyxy",
            dtype=self.compute_dtype,
        )

        y_pred = utils.sort_bounding_boxes(
            y_pred, axis=bounding_box.XYXY.CONFIDENCE
        )

        num_images = tf.shape(y_true)[0]

        iou_thresholds = tf.constant(self.iou_thresholds, dtype=tf.float32)
        class_ids = tf.constant(self.class_ids, dtype=tf.float32)

        num_thresholds = tf.shape(iou_thresholds)[0]
        num_categories = tf.shape(class_ids)[0]

        true_positives_update = tf.zeros_like(self.true_positives)
        ground_truth_boxes_update = tf.zeros_like(self.ground_truth_boxes)

        for img in tf.range(num_images):
            y_true_for_image = utils.get_boxes_for_image(y_true, img)

            y_pred_for_image = utils.get_boxes_for_image(y_pred, img)
            y_pred_for_image = utils.order_by_confidence(y_pred_for_image)

            if self.area_range is not None:
                y_true_for_image = utils.filter_boxes_by_area_range(
                    y_true_for_image, self.area_range[0], self.area_range[1]
                )
                y_pred_for_image = utils.filter_boxes_by_area_range(
                    y_pred_for_image, self.area_range[0], self.area_range[1]
                )

            for k_i in tf.range(num_categories):
                category = class_ids[k_i]

                category_filtered_y_pred = utils.filter_boxes(
                    y_pred_for_image,
                    value=category,
                    axis=bounding_box.XYXY.CLASS,
                )

                detections = category_filtered_y_pred
                if (
                    self.max_detections
                    < tf.shape(category_filtered_y_pred["classes"])[0]
                ):
                    detections = utils.slice(
                        category_filtered_y_pred, self.max_detections
                    )

                ground_truths = utils.filter_boxes(
                    y_true_for_image,
                    value=category,
                    axis=bounding_box.XYXY.CLASS,
                )

                ious = iou_lib.compute_iou(
                    ground_truths["boxes"], detections["boxes"], "yxyx"
                )

                for t_i in tf.range(num_thresholds):
                    threshold = iou_thresholds[t_i]
                    pred_matches = utils.match_boxes(ious, threshold)

                    indices = [t_i, k_i]
                    true_positives = tf.cast(pred_matches != -1, tf.int32)
                    true_positives_sum = tf.math.reduce_sum(
                        true_positives, axis=-1
                    )

                    true_positives_update = tf.tensor_scatter_nd_add(
                        true_positives_update, [indices], [true_positives_sum]
                    )

                ground_truth_boxes_update = tf.tensor_scatter_nd_add(
                    ground_truth_boxes_update,
                    [[k_i]],
                    [tf.cast(tf.shape(ground_truths["classes"])[0], tf.int32)],
                )

        self.true_positives.assign_add(true_positives_update)
        self.ground_truth_boxes.assign_add(ground_truth_boxes_update)

    @tf.function
    def result(self):
        present_values = self.ground_truth_boxes != 0
        n_present_categories = tf.math.reduce_sum(
            tf.cast(present_values, tf.int32), axis=-1
        )
        n_present_categories = tf.cast(n_present_categories, self.dtype)

        if n_present_categories == 0.0:
            return 0.0

        true_positives = tf.cast(self.true_positives, self.dtype)
        ground_truth_boxes = tf.cast(self.ground_truth_boxes, self.dtype)

        recalls = tf.math.divide_no_nan(
            true_positives, ground_truth_boxes[None, :]
        )
        recalls_per_threshold = (
            tf.math.reduce_sum(recalls, axis=-1) / n_present_categories
        )
        return tf.math.reduce_mean(recalls_per_threshold)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_ids": self.class_ids,
                "bounding_box_format": self.bounding_box_format,
                "iou_thresholds": self.iou_thresholds,
                "area_range": self.area_range,
                "max_detections": self.max_detections,
            }
        )
        return config
