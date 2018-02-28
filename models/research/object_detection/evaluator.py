# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Detection model evaluator.

This file provides a generic evaluation method that can be used to evaluate a
DetectionModel.
"""

import logging
import tensorflow as tf

from object_detection import eval_util
from object_detection.core import prefetcher
from object_detection.core import standard_fields as fields
from object_detection.utils import object_detection_evaluation

# A dictionary of metric names to classes that implement the metric. The classes
# in the dictionary must implement
# utils.object_detection_evaluation.DetectionEvaluator interface.
EVAL_METRICS_CLASS_DICT = {
    'pascal_voc_metrics':
        object_detection_evaluation.PascalDetectionEvaluator,
    'weighted_pascal_voc_metrics':
        object_detection_evaluation.WeightedPascalDetectionEvaluator,
    'open_images_metrics':
        object_detection_evaluation.OpenImagesDetectionEvaluator
}


def _extract_prediction_tensors(model,
                                create_input_dict_fn,
                                ignore_groundtruth=False):
  """Restores the model in a tensorflow session.

  Args:
    model: model to perform predictions with.
    create_input_dict_fn: function to create input tensor dictionaries.
    ignore_groundtruth: whether groundtruth should be ignored.

  Returns:
    tensor_dict: A tensor dictionary with evaluations.
  """
  input_dict = create_input_dict_fn()
  prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
  input_dict = prefetch_queue.dequeue()
  original_image = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
  preprocessed_image = model.preprocess(tf.to_float(original_image))
  prediction_dict = model.predict(preprocessed_image)
  detections = model.postprocess(prediction_dict)

  groundtruth = None
  if not ignore_groundtruth:
    groundtruth = {
        fields.InputDataFields.groundtruth_boxes:
            input_dict[fields.InputDataFields.groundtruth_boxes],
        fields.InputDataFields.groundtruth_classes:
            input_dict[fields.InputDataFields.groundtruth_classes],
        fields.InputDataFields.groundtruth_area:
            input_dict[fields.InputDataFields.groundtruth_area],
        fields.InputDataFields.groundtruth_is_crowd:
            input_dict[fields.InputDataFields.groundtruth_is_crowd],
        fields.InputDataFields.groundtruth_difficult:
            input_dict[fields.InputDataFields.groundtruth_difficult]
    }
    if fields.InputDataFields.groundtruth_group_of in input_dict:
      groundtruth[fields.InputDataFields.groundtruth_group_of] = (
          input_dict[fields.InputDataFields.groundtruth_group_of])
    if fields.DetectionResultFields.detection_masks in detections:
      groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
          input_dict[fields.InputDataFields.groundtruth_instance_masks])

  return eval_util.result_dict_for_single_example(
      original_image,
      input_dict[fields.InputDataFields.source_id],
      detections,
      groundtruth,
      class_agnostic=(
          fields.DetectionResultFields.detection_classes not in detections),
      scale_to_absolute=True)


def get_evaluators(eval_config, categories):
  """Returns the evaluator class according to eval_config, valid for categories.

  Args:
    eval_config: evaluation configurations.
    categories: a list of categories to evaluate.
  Returns:
    An list of instances of DetectionEvaluator.

  Raises:
    ValueError: if metric is not in the metric class dictionary.
  """
  eval_metric_fn_key = eval_config.metrics_set
  if eval_metric_fn_key not in EVAL_METRICS_CLASS_DICT:
    raise ValueError('Metric not found: {}'.format(eval_metric_fn_key))
  return [
      EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](
          categories=categories)
  ]


def evaluate(create_input_dict_fn, create_model_fn, eval_config, categories,
             checkpoint_dir, eval_dir):
  """Evaluation function for detection models.

  Args:
    create_input_dict_fn: a function to create a tensor input dictionary.
    create_model_fn: a function that creates a DetectionModel.
    eval_config: a eval_pb2.EvalConfig protobuf.
    categories: a list of category dictionaries. Each dict in the list should
                have an integer 'id' field and string 'name' field.
    checkpoint_dir: directory to load the checkpoints to evaluate from.
    eval_dir: directory to write evaluation metrics summary to.

  Returns:
    metrics: A dictionary containing metric names and values from the latest
      run.
  """

  model = create_model_fn()

  if eval_config.ignore_groundtruth and not eval_config.export_path:
    logging.fatal('If ignore_groundtruth=True then an export_path is '
                  'required. Aborting!!!')

  tensor_dict = _extract_prediction_tensors(
      model=model,
      create_input_dict_fn=create_input_dict_fn,
      ignore_groundtruth=eval_config.ignore_groundtruth)
  
  
  def _process_batch(tensor_dict, sess, batch_index, counters):
    """Evaluates tensors in tensor_dict, visualizing the first K examples.

    This function calls sess.run on tensor_dict, evaluating the original_image
    tensor only on the first K examples and visualizing detections overlaid
    on this original_image.

    Args:
      tensor_dict: a dictionary of tensors
      sess: tensorflow session
      batch_index: the index of the batch amongst all batches in the run.
      counters: a dictionary holding 'success' and 'skipped' fields which can
        be updated to keep track of number of successful and failed runs,
        respectively.  If these fields are not updated, then the success/skipped
        counter values shown at the end of evaluation will be incorrect.

    Returns:
      result_dict: a dictionary of numpy arrays
    """
    try:
      result_dict = sess.run(tensor_dict)
      counters['success'] += 1
    except tf.errors.InvalidArgumentError:
      logging.info('Skipping image')
      counters['skipped'] += 1
      return {}
    global_step = tf.train.global_step(sess, tf.train.get_global_step())
    eval_util.save_values_matrix(sess, result_dict)
    if batch_index < eval_config.num_visualizations:
      tag = 'image-{}'.format(batch_index)
      eval_util.visualize_detection_results(
	  sess,
          result_dict,
          tag,
          global_step,
          categories=categories,
          summary_dir=eval_dir,
          export_dir=eval_config.visualization_export_dir,
          show_groundtruth=eval_config.visualization_export_dir)
#    print (result_dict)
    return result_dict

  variables_to_restore = tf.global_variables()
  global_step = tf.train.get_or_create_global_step()
  variables_to_restore.append(global_step)
  if eval_config.use_moving_averages:
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  def _restore_latest_checkpoint(sess):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)


  #############################################################
  def _save_values_matrix(sess, _process_batch, max_num_prediction):
    if not set([
        'original_image', 'detection_boxes', 'detection_scores',
        'detection_classes'
    ]).issubset(set(result_dict.keys())):
      raise ValueError('result_dict does not contain all expected keys.')
    if show_groundtruth and 'groundtruth_boxes' not in result_dict:
      raise ValueError('If show_groundtruth is enabled, result_dict must contain '
                     'groundtruth_boxes.')
    logging.info('Creating detection visualizations.')
    category_index = label_map_util.create_category_index(categories)

    image = np.squeeze(result_dict['original_image'], axis=0)
    detection_boxes = result_dict['detection_boxes']
    detection_scores = result_dict['detection_scores']
    detection_classes = np.int32((result_dict['detection_classes']))
    groundtruth_classes = np.int32((result_dict['groundtruth_classes']))
    detection_keypoints = result_dict.get('detection_keypoints', None)
    detection_masks = result_dict.get('detection_masks', None)
    groundtruth_boxes_b = result_dict['groundtruth_boxes']
    detection_boxes_2= box_list.BoxList(tf.convert_to_tensor(detection_boxes))
    groundtruth_boxes_2= box_list.BoxList(tf.convert_to_tensor(groundtruth_boxes_b))
    matchGT_matrix = list()
    FP_matrix= list()
    prueba=np.matrix([[0,1],[2,3]])
    if not isinstance(groundtruth_boxes_2, box_list.BoxList):
      raise ValueError('anchors must be an BoxList')
    t = region_similarity_calculator.IouSimilarity()
    compare_tensor = t.compare(detection_boxes_2,groundtruth_boxes_2)
    compare_array = sess.run(compare_tensor)
    len_compare = compare_array.shape
    for i in range(len_compare[1]):
      max_score= 0
      flag= 1
      for j in range(len_compare[0]):
        if len_compare[1]==1:
          IrU = compare_array[j]
        else:
          IrU = compare_array[j][i]
        if IrU>0.6 and detection_scores[j]>max_score:
          max_score= detection_scores[j]
          if flag==1:
            matchGT_matrix.append([detection_classes[j], groundtruth_classes[i], detection_scores[j]])
            flag= 0
          else:
            matchGT_matrix[i]=[detection_classes[j], groundtruth_classes[i], detection_scores[j]]
        if j==99 and max_score<0.6:
          matchGT_matrix[i]=[0, groundtruth_classes[i], max_score]
        if IrU<0.2 and i==1 and np.max(detection_scores[j])>0.5:
          FP_matrix.append([detection_classes[j], 0, detection_scores[j]])

    with open('/home/jesus.molina/matchGT_matrix.csv', 'r') as csv_file:
      newFileReader = csv.reader(csv_file)
      for row in newFileReader:
        matchGT_matrix.append(row)
    #print matchGT_matrix[1][0]

    with open('/home/jesus.molina/matchGT_matrix.csv', 'w') as csv_write:
      writer = csv.writer(csv_write, lineterminator='\n')
      writer.writerows(matchGT_matrix)
    
    with open('/home/jesus.molina/FP_matrix.csv', 'r') as csv_file:
      newFileReader = csv.reader(csv_file)
      for row in newFileReader:
       FP_matrix.append(row)
    #print matchGT_matrix[1][0]

    with open('/home/jesus.molina/FP_matrix.csv', 'w') as csv_write:
      writer = csv.writer(csv_write, lineterminator='\n')
      writer.writerows(FP_matrix) 
    return 1
  #################################################################
  #################################################################
  #################################################################
  Pdd=  _save_values_matrix
  
  metrics = eval_util.repeated_checkpoint_run(
      tensor_dict=tensor_dict,
      summary_dir=eval_dir,
      evaluators=get_evaluators(eval_config, categories),
      batch_processor=_process_batch,
      checkpoint_dirs=[checkpoint_dir],
      variables_to_restore=None,
      restore_fn=_restore_latest_checkpoint,
      num_batches=eval_config.num_examples,
      eval_interval_secs=eval_config.eval_interval_secs,
      max_number_of_evaluations=(1 if eval_config.ignore_groundtruth else
                                 eval_config.max_evals
                                 if eval_config.max_evals else None),
      master=eval_config.eval_master,
      save_graph=eval_config.save_graph,
      save_graph_dir=(eval_dir if eval_config.save_graph else ''))
  
  return metrics 
