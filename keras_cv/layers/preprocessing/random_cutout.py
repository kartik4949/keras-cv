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
import tensorflow.keras.layers as layers
from tensorflow.keras import backend
from tensorflow.python.keras.utils import layer_utils

from keras_cv.utils import fill_utils


class RandomCutout(layers.Layer):
    """Randomly cut out rectangles from images and fill them.

    Args:
        height_factor: a positive float represented as fraction of image height,
            or as absolute height when factor is an integer. When factor is a tuple
            of size 2 it represents lower and upper bound for height. For instance,
            `height_factor=(0.2, 0.3)` results in a height randomly picked in the
            range `[20% of image height, 30% of image height]`, and
            `height_factor=(32, 64)` results in a height picked in the range
            [32, 64]. `height_factor=0.2` results in a height of 20% of image
            height, and `height_factor=32` results in a height of 32.
        width_factor: a positive float represented as fraction of image width,
            or as absolute width when factor is an integer. When factor is a tuple
            of size 2 it represents lower and upper bound for width. For instance,
            `width_factor=(0.2, 0.3)` results in a width randomly picked in the
            range `[20% of image width, 30% of image width]`, and
            `width_factor=(32, 64)` results in a width picked in the range
            [32, 64]. `width_factor=0.2` results in a width of 20% of image
            width, and `width_factor=32` results in a width of 32.
        fill_mode: Pixels inside the patches are filled according to the given
            mode (one of `{"constant", "gaussian_noise"}`).
            - *constant*: Pixels are filled with the same constant value.
            - *gaussian_noise*: Pixels are filled with random gaussian noise.
        fill_value: a float represents the value to be filled inside the patches
            when `fill_mode="constant"`.
        rate: Float between 0 and 1. The probability of augmenting an input.
            Defaults to 1.0.

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_cutout = keras_cv.layers.preprocessing.RandomCutout(0.5, 0.5, rate=1.0)
    augmented_images = random_cutout(images)
    ```
    """

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        rate=1.0,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        layer_utils.validate_string_arg(
            fill_mode,
            allowable_strings=["constant", "gaussian_noise"],
            layer_name="RandomCutout",
            arg_name="fill_mode",
            allow_none=False,
            allow_callables=False,
        )

        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = height_factor
            self.height_upper = height_factor
        if type(self.height_lower) != type(self.height_upper):
            raise ValueError(
                "`height_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.height_lower), type(self.height_upper)
                )
            )
        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                "lower bound, got {}".format(height_factor)
            )
        self._height_is_float = isinstance(self.height_lower, float)
        if self._height_is_float:
            if not self.height_lower >= 0.0 or not self.height_upper <= 1.0:
                raise ValueError(
                    "`height_factor` must have values between [0, 1] "
                    "when is float, got {}".format(height_factor)
                )

        if isinstance(width_factor, (tuple, list)):
            self.width_lower = width_factor[0]
            self.width_upper = width_factor[1]
        else:
            self.width_lower = width_factor
            self.width_upper = width_factor
        if type(self.width_lower) != type(self.width_upper):
            raise ValueError(
                "`width_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.width_lower), type(self.width_upper)
                )
            )
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                "lower bound, got {}".format(width_factor)
            )
        self._width_is_float = isinstance(self.width_lower, float)
        if self._width_is_float:
            if not self.width_lower >= 0.0 or not self.width_upper <= 1.0:
                raise ValueError(
                    "`width_factor` must have values between [0, 1] "
                    "when is float, got {}".format(width_factor)
                )
            self._width_is_float = True

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.rate = rate
        self.seed = seed

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        rate_cond = tf.less(
            tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.rate
        )
        augment_cond = tf.logical_and(rate_cond, training)
        augment = lambda: self._random_cutout(inputs)
        no_augment = lambda: inputs
        return tf.cond(augment_cond, augment, no_augment)

    def _random_cutout(self, inputs):
        """Apply random cutout."""
        center_x, center_y = self._compute_rectangle_position(inputs)
        rectangle_height, rectangle_width = self._compute_rectangle_size(inputs)
        rectangle_fill = self._compute_rectangle_fill(inputs)
        inputs = tf.map_fn(
            lambda x: fill_utils.fill_rectangle(*x),
            (
                inputs,
                center_y,
                center_x,
                rectangle_width // 2,
                rectangle_height // 2,
                rectangle_fill,
            ),
            fn_output_signature=tf.TensorSpec.from_tensor(inputs[0]),
        )
        return inputs

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        center_x = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
            seed=self.seed,
        )
        center_y = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
            seed=self.seed,
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        height = tf.random.uniform(
            [batch_size],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width = tf.random.uniform(
            [batch_size],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )

        if self._height_is_float:
            height = height * tf.cast(image_height, tf.float32)

        if self._width_is_float:
            width = width * tf.cast(image_width, tf.float32)

        height = tf.cast(height, tf.int32)
        width = tf.cast(width, tf.int32)

        height = tf.minimum(height, image_height)
        width = tf.minimum(width, image_width)

        return height, width

    def _compute_rectangle_fill(self, inputs):
        input_shape = tf.shape(inputs)
        if self.fill_mode == "constant":
            fill_value = tf.fill(input_shape, self.fill_value)
        else:
            # gaussian noise
            fill_value = tf.random.normal(input_shape)

        return fill_value

    def get_config(self):
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "rate": self.rate,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
