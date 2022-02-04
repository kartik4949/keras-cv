from keras_cv.metrics.coco.simple_map import COCOMeanAveragePrecision
import numpy as np
import tensorflow as tf

map = COCOMeanAveragePrecision()
# These would match if they were in the area range
y_true = np.array([
        [
            [0, 0, 10, 10, 1], [0, 0, 10, 10, 2]
        ],
        [
            [0, 0, 10, 10, 1], [0, 0, 10, 10, 2]
        ]
    ]
).astype(np.float32)
y_pred = np.array([
    [
        [0, 0, 10, 10, 1, 0.7],
        [-1, -1, -1, -1, -1, -1]
    ],
    [
        [0, 0, 10, 10, 1, 0.5],
        [0, 0, 10, 10, 1, 0.5]
    ]
]).astype(
    np.float32
)

map.update_state(tf.constant(y_true), tf.constant(y_pred))

result = map.result()
print('Result', result)