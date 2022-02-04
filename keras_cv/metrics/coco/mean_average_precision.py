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
        category_ids,
        recall_thresholds=None,
        iou_thresholds=None,
        area_range=(0, 1e9**2),
        max_detections=100,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize parameter values
        self._user_iou_thresholds = iou_thresholds or [
            x / 100.0 for x in range(50, 100, 5)
        ]
        self.iou_thresholds = self._add_constant_weight(
            "iou_thresholds", self._user_iou_thresholds
        )
        # TODO(lukewood): support inference of category_ids based on update_state.
        self.category_ids = self._add_constant_weight("category_ids", category_ids)

        self.area_range = area_range
        self.max_detections = max_detections

        # Initialize result counters
        self.num_thresholds = len(self._user_iou_thresholds)
        self.num_categories = len(category_ids)

        self.true_positives = self.add_weight(
            name="true_positives",
            shape=(self.num_thresholds, self.num_categories),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.false_positives = self.add_weight(
            name="false_positives",
            shape=(self.num_thresholds, self.num_categories),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.ground_truth_boxes = self.add_weight(
            name="ground_truth_boxes",
            shape=(self.num_categories,),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        recall_thresholds = recall_thresholds or [x/100. for x in range(0, 101)]
        self.recall_thresholds = self._add_constant_weight('recall_thresholds', recall_thresholds)
        self.num_recall_thresholds = len(recall_thresholds)
    
    def update_state(self, y_true, y_pred):
        num_images = tf.shape(y_true)[0]

        num_thresholds = self.num_thresholds
        num_categories = self.num_categories
        y_pred = utils.sort_bboxes(y_pred, axis=bbox.CONFIDENCE)

        pad_size = tf.shape(y_pred)
        dtm = tf.TensorArray(tf.float32, size=num_images*num_categories*num_thresholds, dynamic_size=False)
        ground_truths = tf.TensorArray(tf.float32, size=num_images*num_categories, dynamic_size=False)

        for img in tf.range(num_images):
            sentinel_filtered_y_true = utils.filter_out_sentinels(y_true[img])
            sentinel_filtered_y_pred = utils.filter_out_sentinels(y_pred[img])

            area_filtered_y_true = utils.filter_boxes_by_area_range(
                sentinel_filtered_y_true, self.area_range[0], self.area_range[1]
            )
            area_filtered_y_pred = utils.filter_boxes_by_area_range(
                sentinel_filtered_y_pred, self.area_range[0], self.area_range[1]
            )

            for k_i in tf.range(num_categories):
                category = self.category_ids[k_i]

                category_filtered_y_pred = utils.filter_boxes(
                    area_filtered_y_pred, value=category, axis=bbox.CLASS
                )

                detections = category_filtered_y_pred
                if self.max_detections < tf.shape(category_filtered_y_pred)[0]:
                    detections = category_filtered_y_pred[:self.max_detections]

                ground_truths = utils.filter_boxes(
                    area_filtered_y_true, value=category, axis=bbox.CLASS
                )
                ground_truths = ground_truths.write(k_i + (img*num_categories), tf.shape(ground_truths)[0])

                ious = iou_lib.compute_ious_for_image(ground_truths, detections)

                for t_i in tf.range(num_thresholds):
                    threshold = self.iou_thresholds[t_i]
                    pred_matches = self._match_boxes(
                        ground_truths, detections, threshold, ious
                    )
                    index = img * (num_categories*num_thresholds) + (k_i * num_thresholds) + t_i 
                    dtm = dtm.write(index, pred_matches)

        dtm = tf.reshape(dtm.stack(), (num_images, num_categories, num_thresholds))
        ground_truths = tf.reshape(dtm.stack(), (num_images, num_categories))

        dtm = tf.reduce_sum(dtm, axis=0)
        ground_truths = tf.reduce_sum(ground_truths, axis=0)

        rcs = []
        prs = []

        """
        Notes:
            in order to support MaP tracking across batches we need the following:
            we will need to track which bounding boxes were classified as true positives,
            which were false positives, their confidence scores, and track all
            of this information across categories & IoU thresholds.

            Then, we can use this information in `result()` to compute the exact
            MaP value for the given set of bounding boxes.
        """
        for k_i in tf.range(num_categories):
            dts = dtm[k_i]
            gts = ground_truths[k_i]

            tps = dts != 0
            fps = dts == 0

            tp_sums = tf.math.cumsum(tps, axis=0)
            fp_sums = tf.math.cumsum(fps, axis=0)
            
            for idx in tf.range(tf.shape(tp_sums)[0]):
                tp = tp_sums[idx]
                fp = fp_sums[idx]
                rc = tf.math.divide_no_nan(tp, gts)
                pr = tf.math.divide_no_nan(tp, (fp + tp))
                
