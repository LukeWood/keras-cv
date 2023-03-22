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

import functools

import numpy as np
import tensorflow as tf

import keras_cv
import keras_cv.visualization.draw_bounding_boxes
from keras_cv import utils
from keras_cv.utils import assert_cv2_installed
from keras_cv.utils import assert_matplotlib_installed
from keras_cv.visualization.plot_gallery import plot_gallery

try:
    from matplotlib import patches
except:
    patches = None


def plot_bounding_box_gallery(
    images,
    value_range,
    bounding_box_format,
    pred_color=(255, 235, 59),
    true_color=(0, 188, 212),
    y_true=None,
    y_pred=None,
    thickness=2,
    font_scale=1.0,
    text_thickness=None,
    class_mapping=None,
    ground_truth_mapping=None,
    prediction_mapping=None,
    legend=False,
    legend_handles=None,
    rows=3,
    cols=3,
    **kwargs
):
    """plots a gallery of images with corresponding bounding box annotations

    ![Example bounding box gallery](https://i.imgur.com/Fy7kMnP.png)

    Args:
        images: a Tensor or NumPy array containing images to show in the gallery.
        value_range: value range of the images.
        bounding_box_format: the bounding_box_format  the provided bounding boxes are
            in.
        y_true: a KerasCV bounding box dictionary representing the ground truth bounding
            boxes.
        y_pred: a KerasCV bounding box dictionary representing the predicted truth
            bounding boxes.
        pred_color: three element tuple representing the color to use for plotting
            predicted bounding boxes.
        true_color: three element tuple representing the color to use for plotting
            true bounding boxes.
        class_mapping: (Optional) class mapping from class IDs to strings
        ground_truth_mapping:  (Optional) class mapping from class IDs to strings,
            defaults to `class_mapping`
        prediction_mapping:  (Optional) class mapping from class IDs to strings,
            defaults to `class_mapping`
        thickness: (Optional) thickness for the box and text labels.  Defaults to 2.
        text_thickness: (Optional) the thickness for the text, defaults to `1.0`.
        font_scale: (Optional) font size to draw bounding boxes in.
        legend: Whether or not to create a legend with the specified colors for `y_true`
            and `y_pred`.  Defaults to False.
        kwargs: keyword arguments to propagate to
            `keras_cv.visualization.gallery_show()`.
    """
    assert_matplotlib_installed("plot_bounding_box_gallery")
    assert_cv2_installed("plot_bounding_box_gallery")
    if isinstance(images, tf.data.Dataset):
        raise ValueError(
            "Currently `keras_cv.visualization.plot_bounding_box_gallery()` does "
            "not support `tf.data.Datasets`. Please pass a single batch, i.e.:"
            "`images, boxes = next(iter(ds.take(1)))`."
        )
    prediction_mapping = prediction_mapping or class_mapping
    ground_truth_mapping = ground_truth_mapping or class_mapping

    plotted_images = utils.to_numpy(images)

    draw_fn = functools.partial(
        keras_cv.visualization.draw_bounding_boxes.draw_bounding_boxes,
        bounding_box_format="xyxy",
        thickness=thickness,
        text_thickness=text_thickness,
        font_scale=font_scale,
    )

    if y_true is not None:
        y_true = y_true.copy()
        y_true["boxes"] = utils.to_numpy(y_true["boxes"])
        y_true["classes"] = utils.to_numpy(y_true["classes"])
        y_true = keras_cv.bounding_box.convert_format(
            y_true, images=images, source=bounding_box_format, target="xyxy"
        )
        plotted_images = draw_fn(
            plotted_images,
            y_true,
            true_color,
            class_mapping=ground_truth_mapping,
        )

    if y_pred is not None:
        y_pred = y_pred.copy()
        y_pred["boxes"] = utils.to_numpy(y_pred["boxes"])
        y_pred["classes"] = utils.to_numpy(y_pred["classes"])
        y_pred = keras_cv.bounding_box.convert_format(
            y_pred, images=images, source=bounding_box_format, target="xyxy"
        )
        plotted_images = draw_fn(
            plotted_images, y_pred, pred_color, class_mapping=prediction_mapping
        )

    if legend:
        if legend_handles:
            raise ValueError(
                "Only pass `legend` OR `legend_handles` to "
                "`luketils.visualization.plot_bounding_box_gallery()`."
            )
        legend_handles = [
            patches.Patch(
                color=np.array(true_color) / 255.0,
                label="Ground Truth",
            ),
            patches.Patch(
                color=np.array(pred_color) / 255.0,
                label="Prediction",
            ),
        ]

    plot_gallery(
        plotted_images,
        value_range,
        legend_handles=legend_handles,
        rows=rows,
        cols=cols,
        **kwargs
    )
