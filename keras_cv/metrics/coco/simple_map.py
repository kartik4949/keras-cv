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
from numpy import pad
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv.metrics.coco import iou as iou_lib
from keras_cv.metrics.coco import utils
from keras_cv.utils import bbox



class COCOMeanAveragePrecision(tf.keras.metrics.Metric):
    """COCOMeanAveragePrecision computes MaP.

    Args:
        iou_thresholds: defaults to [0.5:0.05:0.95].
        category_ids: no default, users must provide.
        area_range: area range to consider bounding boxes in. Defaults to all.
        max_detections: number of maximum detections a model is allowed to make.
        recall_thresholds: List of floats.  Defaults to [0:.01:1].
    """
    def __init__(self, 
        category_id=1,
        recall_thresholds=None,
        iou_threshold=0.5,
        area_range=(0, 1e9**2),
        max_detections=100,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize parameter values
        self.iou_threshold = iou_threshold
        self.area_range = area_range
        self.max_detections = max_detections
        self.category_id = category_id
        self.recall_thresholds = recall_thresholds
        self.dt_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.dtm =  tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        self.gts = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
    
    def reset_state(self):
        self.dt_scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        self.dtm =  tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
        self.gts = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
    
    def update_state(self, y_true, y_pred):
        num_images = tf.shape(y_true)[0]

        y_pred = utils.sort_bboxes(y_pred, axis=bbox.CONFIDENCE)

        pad_to_shape = tf.shape(y_pred)[0]
        for img in tf.range(num_images):
            ground_truths = utils.filter_out_sentinels(y_true[img])
            detections = utils.filter_out_sentinels(y_pred[img])
            ground_truths = utils.filter_boxes_by_area_range(
                ground_truths, self.area_range[0], self.area_range[1]
            )
            detections = utils.filter_boxes_by_area_range(
                detections, self.area_range[0], self.area_range[1]
            )
            ground_truths = utils.filter_boxes(
                ground_truths, value=self.category_id, axis=bbox.CLASS
            )
            detections = utils.filter_boxes(
                detections, value=self.category_id, axis=bbox.CLASS
            )

            ious = iou_lib.compute_ious_for_image(ground_truths, detections)
            pred_matches = utils.match_boxes(
                ground_truths, detections, self.iou_threshold, ious
            )

            self.dt_scores = self.dt_scores.write(img, 
                bbox.pad_bbox_batch_to_shape(detections[:, bbox.CONFIDENCE], (pad_to_shape,))
            )
            self.dtm = self.dtm.write(img, bbox.pad_bbox_batch_to_shape(pred_matches, (pad_to_shape,)))
            self.gts = self.gts.write(img, tf.shape(ground_truths)[0])

    def result(self):
        dt_scores = self.dt_scores.stack()
        dt_scores = tf.reshape(dt_scores, (-1,))
        dtm = self.dtm.stack()
        dtm = tf.reshape(dtm, (-1,))

        gts = self.gts.stack()
        gts = tf.math.reduce_sum(gts)

        indices = tf.argsort(dt_scores, direction='DESCENDING')
        dtm = tf.gather(dtm, indices)
        tf.print(dtm)
        dtm = tf.gather_nd(dtm, tf.where(dtm != -1))

        tps = tf.cast(dtm != -2, tf.float32)
        fps = tf.cast(dtm == -2, tf.float32)

        tp_sum = tf.cumsum(tps, axis=-1)
        fp_sum = tf.cumsum(fps, axis=-1)
        precision_result = tf.TensorArray(tf.float32, size=len(self.recall_thresholds))
        for tp_idx in tf.range(tf.shape(tp_sum)[0]):
            tp = tp_sum[tp_idx]
            fp = fp_sum[tp_idx]
            
            rc = tp / gts
            pr = tp / (fp + tp)

            inds = tf.search_sorted(rc, tf.constant(self.recall_thresholds), side='left')
            for ri in tf.range(len(self.recall_thresholds)):
                pi = inds[ri]
                precision_result = precision_result.write(ri, pi)
        return tf.math.reduce_mean(precision_result.stack())
