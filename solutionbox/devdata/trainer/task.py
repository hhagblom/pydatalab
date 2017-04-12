# Copyright 2017 Google Inc. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import re
import sys

# run via
# python -m trainer/task --train-data-paths train_csv_data.csv  --eval-data-paths eval_csv_data.csv  --job-dir TOUT --preprocess-output-dir x --transforms-file y --model-type dnn_classification --top-n 3


#from . import util
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.saved import saved_transform_io



from tensorflow.python.ops import variables
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.python.training import saver
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.client import session as tf_session
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_def_utils



from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)


def _recursive_copy(src_dir, dest_dir):
  """Copy the contents of src_dir into the folder dest_dir.
  Args:
    src_dir: gsc or local path.
    dest_dir: gcs or local path.
  When called, dest_dir should exist.
  """

  file_io.recursive_create_dir(dest_dir)
  for file_name in file_io.list_directory(src_dir):
    old_path = os.path.join(src_dir, file_name)
    new_path = os.path.join(dest_dir, file_name)

    if file_io.is_directory(old_path):
      _recursive_copy(old_path, new_path)
    else:
      file_io.copy(old_path, new_path, overwrite=True)


def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))

def get_reader_input_fn(data_paths, batch_size, shuffle, num_epochs=None):
  """Builds input layer for training."""
  transformed_metadata = metadata_io.read_metadata(
        '../tfpreout/transformed_metadata')

  #print('tf metadata')
  #print(transformed_metadata)
  #print(dir(transformed_metadata))
  #print(dir(transformed_metadata.schema))
  return input_fn_maker.build_training_input_fn(
      metadata=transformed_metadata,
      file_pattern=data_paths,
      training_batch_size=batch_size,
      label_keys=['targetex'],
      reader=gzip_reader_fn,
      key_feature_name='keyex',
      reader_num_threads=4,
      queue_capacity=batch_size * 2,
      randomize_input=shuffle,
      num_epochs=num_epochs)

def get_estimator(train_root, args):

  s1 = tf.contrib.layers.sparse_column_with_integerized_feature('str1ex', bucket_size=8+1) # bucket_size = vocab_size + unknown label
  s2 = tf.contrib.layers.sparse_column_with_integerized_feature('str2ex', bucket_size=7+1)
  s3 = tf.contrib.layers.sparse_column_with_integerized_feature('str3ex', bucket_size=7+1)

  feature_columns = [
      tf.contrib.layers.real_valued_column('num1ex', dimension=1),
      tf.contrib.layers.real_valued_column('num2ex', dimension=1),
      tf.contrib.layers.real_valued_column('num3ex', dimension=1),
      tf.contrib.layers.embedding_column(s1, 2),
      tf.contrib.layers.embedding_column(s2, 2),
      tf.contrib.layers.embedding_column(s3, 2),
  ]
  config = tf.contrib.learn.RunConfig(
      save_checkpoints_secs=args.save_checkpoints_secs)
  estimator = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 8, 5],
        n_classes=3,
        config=config,
        model_dir=train_root,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  return estimator


def make_output_tensors(input_ops, model_fn_ops, keep_target=True):


  #print('make_output_tensors')
  #print('input_ops.features', input_ops.features)
  #print('input_ops.labels', input_ops.labels)
  #print('input_ops.default_inputs', input_ops.default_inputs)
  #print(model_fn_ops)
  #print(model_fn_ops.predictions)
  top_n=3

  outputs = {}
  outputs['keyex'] = tf.squeeze(input_ops.features['keyex'])
  if keep_target:
      outputs['target_from_inputex'] = tf.squeeze(input_ops.features['targetex'])

  # TODO(brandondutra): get the score of the target label too.
  probabilities = model_fn_ops.predictions['probabilities']

  # get top k labels and their scores.
  (top_k_values, top_k_indices) = tf.nn.top_k(probabilities, k=top_n)
  #top_k_labels = table.lookup(tf.to_int64(top_k_indices))
  top_k_labels = top_k_indices

  # Write the top_k values using 2*top_k columns.
  num_digits = int(math.ceil(math.log(top_n, 10)))
  if num_digits == 0:
    num_digits = 1
  for i in range(0, top_n):
    # Pad i based on the size of k. So if k = 100, i = 23 -> i = '023'. This
    # makes sorting the columns easy.
    padded_i = str(i + 1).zfill(num_digits)

    label_alias = 'top_n_label_%s' % padded_i

    label_tensor_name = (tf.squeeze(
        tf.slice(top_k_labels, [0, i], [tf.shape(top_k_labels)[0], 1])))

    score_alias = 'top_n_score_%s' % padded_i

    score_tensor_name = (tf.squeeze(
        tf.slice(top_k_values,
                 [0, i],
                 [tf.shape(top_k_values)[0], 1])))

    outputs.update({label_alias: label_tensor_name,
                    score_alias: score_tensor_name})

  return outputs




def my_make_export_strategy(serving_input_fn, keep_target, job_dir):
  def export_fn(estimator, export_dir_base, checkpoint_path=None, eval_result=None):
    with ops.Graph().as_default() as g:
      contrib_variables.create_global_step(g)

      input_ops = serving_input_fn()
      model_fn_ops = estimator._call_model_fn(input_ops.features,
                                              None,
                                              model_fn_lib.ModeKeys.INFER)
      output_fetch_tensors = make_output_tensors(
          input_ops=input_ops,
          model_fn_ops=model_fn_ops,
          keep_target=keep_target)

      signature_def_map = {
        'serving_default': signature_def_utils.predict_signature_def(input_ops.default_inputs,
                                                                     output_fetch_tensors)
      }

      if not checkpoint_path:
        # Locate the latest checkpoint
        checkpoint_path = saver.latest_checkpoint(estimator._model_dir)
      if not checkpoint_path:
        raise NotFittedError("Couldn't find trained model at %s."
                             % estimator._model_dir)

      export_dir = saved_model_export_utils.get_timestamped_export_dir(
          export_dir_base)

      with tf_session.Session('') as session:
        # variables.initialize_local_variables()
        variables.local_variables_initializer()
        data_flow_ops.tables_initializer()
        saver_for_restore = saver.Saver(
            variables.global_variables(),
            sharded=True)
        saver_for_restore.restore(session, checkpoint_path)

        init_op = control_flow_ops.group(
            variables.local_variables_initializer(),
            data_flow_ops.tables_initializer())

        # Perform the export
        builder = saved_model_builder.SavedModelBuilder(export_dir)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=ops.get_collection(
                ops.GraphKeys.ASSET_FILEPATHS),
            legacy_init_op=init_op)
        builder.save(False)

      # Add the extra assets
      #if assets_extra:
      #....

    # only keep the last 3 models
    saved_model_export_utils.garbage_collect_exports(
        export_dir_base,
        exports_to_keep=3)

    # save the last model to the model folder.
    # export_dir_base = A/B/intermediate_models/
    if keep_target:
      final_dir = os.path.join(job_dir, 'evaluation_model')
    else:
      final_dir = os.path.join(job_dir, 'model')
    if file_io.is_directory(final_dir):
      file_io.delete_recursively(final_dir)
    file_io.recursive_create_dir(final_dir)
    _recursive_copy(export_dir, final_dir)

    return export_dir

  if keep_target:
    intermediate_dir = 'intermediate_evaluation_models'
  else:
    intermediate_dir = 'intermediate_prediction_models'

  return export_strategy.ExportStrategy(intermediate_dir, export_fn)  

def get_experiment_fn(args):
  """Builds the experiment function for learn_runner.run.

  Args:
    args: the command line args

  Returns:
    A function that returns a tf.learn experiment object.
  """

  def get_experiment(output_dir):
    # Merge schema, input features, and transforms.
    # Get the model to train.
    estimator = get_estimator(output_dir, args)

    transformed_metadata = metadata_io.read_metadata(
        '../tfpreout/transformed_metadata')
    raw_metadata = metadata_io.read_metadata('../tfpreout/raw_metadata')
    #serving_input_fn = (
    #    input_fn_maker.build_default_transforming_serving_input_fn(  #input json string
    #        raw_metadata,
    #        '../tfpreout/transform_fn',
    #        raw_label_keys=['target']))    
    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(  # input is tf.example string
            raw_metadata,
            '../tfpreout/transform_fn',
            raw_label_keys=['target']))
    #export_strategy_notarget = tf.contrib.learn.utils.make_export_strategy(
    #    serving_input_fn, exports_to_keep=5,
    #    default_output_alternative_key=None)
    export_strategy_target = my_make_export_strategy(serving_input_fn, keep_target=True, job_dir=args.job_dir)
    export_strategy_notarget = my_make_export_strategy(serving_input_fn, keep_target=False, job_dir=args.job_dir)
    #export_strategy = my_make_export_strategy(serving_input_fn, keep_target=True)


    input_reader_for_train = get_reader_input_fn(
        data_paths=args.train_data_paths,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_epochs=args.num_epochs)

    input_reader_for_eval = get_reader_input_fn(
        data_paths=args.eval_data_paths,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_epochs=1)

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=input_reader_for_train,
        eval_input_fn=input_reader_for_eval,
        train_steps=args.max_steps,
        export_strategies=[export_strategy_target, export_strategy_notarget],
        min_eval_frequency=args.min_eval_frequency,
        eval_steps=None,
    )

  # Return a function to create an Experiment.
  return get_experiment

 #python -m trainer/t\--job-dir TOUT --preprocess-output-dir x --transforms-file y --model-type dnn_classification --top-n 3

def parse_arguments(argv):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser(
      description=('Train a regression or classification model. Note that if '
                   'using a DNN model, --layer-size1=NUM, --layer-size2=NUM, '
                   'should be used. '))

  # I/O file parameters
  parser.add_argument('--train-data-paths', type=str, action='append',
                      default='../tfpreout/features_train-00000-of-00001.tfrecord.gz')
  parser.add_argument('--eval-data-paths', type=str, action='append',
                      default='../tfpreout/features_eval-00000-of-00001.tfrecord.gz')
  parser.add_argument('--job-dir', type=str, default='../TOUT')
  parser.add_argument('--preprocess-output-dir',
                      type=str,
                      required=False,
                      help=('Output folder of preprocessing. Should contain the'
                            ' schema file, and numerical stats and vocab files.'
                            ' Path must be on GCS if running'
                            ' cloud training.'))
  parser.add_argument('--transforms-file',
                      type=str,
                      required=False,
                      help=('File describing the the transforms to apply on '
                            'each column'))

  # HP parameters
  parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='tf.train.AdamOptimizer learning rate')
  parser.add_argument('--epsilon', type=float, default=0.0005,
                      help='tf.train.AdamOptimizer epsilon')
  # --layer_size See below

  # Model problems
  parser.add_argument('--model-type',
                      choices=['linear_classification', 'linear_regression',
                               'dnn_classification', 'dnn_regression'],
                      required=False)
  parser.add_argument('--top-n',
                      type=int,
                      default=1,
                      help=('For classification problems, the output graph '
                            'will contain the labels and scores for the top '
                            'n classes.'))
  # Training input parameters
  parser.add_argument('--max-steps', type=int, default=250,
                      help='Maximum number of training steps to perform.')
  parser.add_argument('--num-epochs',
                      type=int,
                      help=('Maximum number of training data epochs on which '
                            'to train. If both --max-steps and --num-epochs '
                            'are specified, the training job will run for '
                            '--max-steps or --num-epochs, whichever occurs '
                            'first. If unspecified will run for --max-steps.'))
  parser.add_argument('--train-batch-size', type=int, default=1000)
  parser.add_argument('--eval-batch-size', type=int, default=1000)
  parser.add_argument('--min-eval-frequency', type=int, default=100,
                      help=('Minimum number of training steps between '
                            'evaluations'))

  # other parameters
  parser.add_argument('--save-checkpoints-secs', type=int, default=600,
                      help=('How often the model should be checkpointed/saved '
                            'in seconds'))

  args, remaining_args = parser.parse_known_args(args=argv[1:])

  # All HP parambeters must be unique, so we need to support an unknown number
  # of --layer_size1=10 --layer_size2=10 ...
  # Look at remaining_args for layer_size\d+ to get the layer info.

  # Get number of layers
  pattern = re.compile('layer-size(\d+)')
  num_layers = 0
  for other_arg in remaining_args:
    match = re.search(pattern, other_arg)
    if match:
      num_layers = max(num_layers, int(match.group(1)))

  # Build a new parser so we catch unknown args and missing layer_sizes.
  parser = argparse.ArgumentParser()
  for i in range(num_layers):
    parser.add_argument('--layer-size%s' % str(i + 1), type=int, required=True)

  layer_args = vars(parser.parse_args(args=remaining_args))
  layer_sizes = []
  for i in range(num_layers):
    key = 'layer_size%s' % str(i + 1)
    layer_sizes.append(layer_args[key])

  assert len(layer_sizes) == num_layers
  args.layer_sizes = layer_sizes

  return args


def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)


if __name__ == '__main__':
  main()
