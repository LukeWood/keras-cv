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

import os

import pytest
import tensorflow as tf
from tensorflow.keras import optimizers

import keras_cv
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)


class OldRetinaNetIdenticalTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        tf.config.set_soft_device_placement(False)
        yield
        # Reset soft device placement to not interfere with other unit test files
        tf.config.set_soft_device_placement(True)
        tf.keras.backend.clear_session()

    def test_retina_net_forward(self):
        retina_net = keras_cv.models.RetinaNet(
            num_classes=20,
            bounding_box_format="xywh",
        )
        retina_net_old = keras_cv.models.object_detection.old_retina_net.OldRetinaNet(
            classes=20,
            bounding_box_format="xywh",
        )

        images = tf.random.uniform((1, 640, 640, 3))
        rno1, rno2 = retina_net(images)
        orno1, orno2 = retina_net_old(images)

        self.assertEqual(rno1.shape, orno1.shape)
        self.assertEqual(rno2.shape, orno2.shape)

        self.assertAllEqual(rno1, orno1)
        self.assertAllEqual(rno2, orno2)
