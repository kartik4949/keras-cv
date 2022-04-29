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
"""A demo for visualization of segmentation masks on flower
dataset i.e `oxford_flowers102`.

Demonstrate visualization of centered rectangle masks on batch
of flower images using `draw_segmentation` utility of keras_cv.
Plot the figure using `matplotlib`
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv.visualization import draw_segmentation

IMG_SIZE = (224, 224)
BATCH_SIZE = 2
MASK_SIZE = 24


def get_image(data):
    image = data["image"]
    image = tf.image.resize(image, IMG_SIZE)
    return image


def main():
    data = tfds.load("oxford_flowers102")
    train_ds = data["train"]

    # get a batch of resized images.
    train_ds = train_ds.map(lambda x: get_image(x)).batch(BATCH_SIZE)
    batch_images = next(iter(train_ds))
    batch_images = tf.cast(batch_images, tf.uint8)

    # create center rectangle mask.
    mask = tf.ones((MASK_SIZE, MASK_SIZE), tf.int32)
    mask_pad = int((IMG_SIZE[0] - MASK_SIZE) / 2)
    paddings = tf.constant([[mask_pad] * 2] * 2)
    mask = tf.cast(tf.pad(mask, paddings, "CONSTANT"), dtype=tf.uint8)
    mask = tf.stack([mask] * BATCH_SIZE, axis=0)

    # draw segmentation on batch of images.
    masked_images = draw_segmentation(batch_images, mask, color="green")

    # show masked images.
    f, axarr = plt.subplots(BATCH_SIZE, 1)
    for i, mask_image in enumerate(masked_images):
        axarr[i].imshow(mask_image.numpy().astype("uint8"))
    plt.show()


if __name__ == "__main__":
    main()
