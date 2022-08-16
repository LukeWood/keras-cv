import tensorflow as tf
from absl.testing import parameterized

from keras_cv import layers as cv_layers


class AnchorGeneratorTest(tf.test.TestCase, parameterized.TestCase):
    def test_output_dictionary_keys(self):
        # TODO(lukewood): write and fix this.
        pass

    def test_relative_generation(self):
        strides = [8, 16, 32]
        scales = [1, 1.2599210498948732, 1.5874010519681994]
        sizes = [32.0, 64.0, 128.0]
        aspect_ratios = [0.5, 1.0, 2.0]

        image = tf.random.uniform((512, 512, 3))
        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="rel_yxyx",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
            clip_boxes=True,
        )
        boxes = anchor_generator(image)
        boxes = tf.concat(list(boxes.values()), axis=0)
        self.assertAllLessEqual(boxes, 1.5)
        self.assertAllGreaterEqual(boxes, -0.50)

    # Constructor parameter validations
    @parameterized.named_parameters(
        ("unequal_lists", [0, 1, 2], [1]),
        ("unequal_levels_dicts", {"level_1": [0, 1, 2]}, {"1": [0, 1, 2]}),
    )
    def test_raises_when_strides_not_equal_to_sizes(self, sizes, strides):
        with self.assertRaises(ValueError):
            cv_layers.AnchorGenerator(
                bounding_box_format="xyxy",
                sizes=sizes,
                strides=strides,
                aspect_ratios=[3 / 4, 1, 4 / 3],
                scales=[0.5, 1.0, 1.5],
            )

    def test_validate_same_level_keys_for_sizes(self):
        aspect_ratios = [0.5, 1.0, 2.0]
        scales = {"l1": 1.0}
        strides = {"l1": [5, 10, 100]}
        sizes = {"l2": [32.0, 64.0, 128.0]}
        with self.assertRaisesRegexp(
            ValueError, "Expected `sizes` to contain the same keys as"
        ):
            anchor_generator = cv_layers.AnchorGenerator(
                bounding_box_format="yxyx",
                sizes=sizes,
                aspect_ratios=aspect_ratios,
                scales=scales,
                strides=strides,
            )

    def test_validate_sizes_and_strides(self):
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        aspect_ratios = [0.5, 1.0, 2.0]

        strides = [5, 10, 100]
        sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
        with self.assertRaisesRegexp(
            ValueError, "Expected `sizes` and `strides` to be of the same length."
        ):
            anchor_generator = cv_layers.AnchorGenerator(
                bounding_box_format="yxyx",
                sizes=sizes,
                aspect_ratios=aspect_ratios,
                scales=scales,
                strides=strides,
            )
        scales = {"l1": 1.0}
        strides = {"l1": strides}
        sizes = {"l1": sizes}
        with self.assertRaisesRegexp(ValueError, "At level 'l1'"):
            anchor_generator = cv_layers.AnchorGenerator(
                bounding_box_format="yxyx",
                sizes=sizes,
                aspect_ratios=aspect_ratios,
                scales=scales,
                strides=strides,
            )

    def test_output_shapes(self):
        strides = [2**i for i in range(3, 8)]
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        aspect_ratios = [0.5, 1.0, 2.0]

        image = tf.random.uniform((512, 512, 3))
        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="yxyx",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )
        boxes = anchor_generator(image)
        boxes = tf.concat(list(boxes.values()), axis=0)
        self.assertEqual(boxes.shape, [49104, 4])
