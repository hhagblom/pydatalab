# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sample for Criteo dataset can be run as a wide or deep model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import json
import math
import os
import sys

import tensorflow as tf

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.contrib.learn.python.learn import learn_runner

from tensorflow.python.framework import dtypes

KEY_FEATURE_COLUMN = 'key'
TARGET_FEATURE_COLUMN = 'target'


def create_parser():
  """Initialize command line parser using arparse.

  Returns:
    An argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_data_paths', default='exout/features_train-*')
  parser.add_argument(
      '--eval_data_paths', default='exout/features_test-*')
  parser.add_argument('--output_path', default='TOUT')
  # The following three parameters are required for tf.Transform.
  parser.add_argument('--raw_metadata_path', default='pout/raw_metadata')
  parser.add_argument('--transformed_metadata_path', default='pout/transformed_metadata')
  parser.add_argument('--transform_savedmodel', default='pout/transform_fn')
  parser.add_argument(
      '--batch_size',
      help='Number of input records used per batch',
      default=10,
      type=int)
  parser.add_argument(
      '--train_steps', help='Number of training steps to perform.', type=int)
  parser.add_argument(
      '--eval_steps',
      help='Number of evaluation steps to perform.',
      type=int,
      default=10)
  return parser


def feature_columns():
  """Return the feature columns with their names and types."""
  result = []

  vocab_size = 7178 + 1



  sparse_id = tf.contrib.layers.sparse_column_with_integerized_feature('text_ids', vocab_size, combiner='sum')
  sparse = tf.contrib.layers.weighted_sparse_column(sparse_id,
                           'text_weights',
                           dtype=dtypes.float32)
  column = sparse
  #column = tf.contrib.layers.weighted_sparse_column(sparse, 'counts', dtypes.float32)
  
  result.append(column)

  return result


def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def get_transformed_reader_input_fn(transformed_metadata,
                                    transformed_data_paths,
                                    batch_size,
                                    mode):
  """Wrap the get input features function to provide the runtime arguments."""
  return input_fn_maker.build_training_input_fn(
      metadata=transformed_metadata,
      file_pattern=transformed_data_paths,
      training_batch_size=batch_size,
      label_keys=[TARGET_FEATURE_COLUMN],
      reader=gzip_reader_fn,
      key_feature_name=KEY_FEATURE_COLUMN,
      reader_num_threads=4,
      queue_capacity=batch_size * 2,
      randomize_input=(mode != tf.contrib.learn.ModeKeys.EVAL),
      num_epochs=(1 if mode == tf.contrib.learn.ModeKeys.EVAL else None))


def get_experiment_fn(args):
  """Wrap the get experiment function to provide the runtime arguments."""
  

  def get_experiment(output_dir):
    """Function that creates an experiment http://goo.gl/HcKHlT.

    Args:
      output_dir: The directory where the training output should be written.
    Returns:
      A `tf.contrib.learn.Experiment`.
    """

    columns = feature_columns()

    runconfig = tf.contrib.learn.RunConfig()
    model_dir = os.path.join(output_dir, 'model')

    #estimator = tf.contrib.learn.DNNClassifier(
    #    feature_columns=columns,
    #    n_classes=2,
    #    hidden_units=[10, 3],
    #    config=runconfig,
    #    model_dir=model_dir,
    #    optimizer=tf.train.AdamOptimizer()
    #)


    #optimizer = tf.contrib.linear_optimizer.SDCAOptimizer(
    #          example_id_column=KEY_FEATURE_COLUMN,
    #          symmetric_l2_regularization=10)
    
    optimizer = tf.train.AdamOptimizer()
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=columns,
        n_classes=20,
        config=runconfig,
        model_dir=model_dir,
        optimizer=optimizer
    )    

    transformed_metadata = metadata_io.read_metadata(
        args.transformed_metadata_path)
    raw_metadata = metadata_io.read_metadata(args.raw_metadata_path)
    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata,
            args.transform_savedmodel,
            raw_label_keys=[TARGET_FEATURE_COLUMN]))
    export_strategy = tf.contrib.learn.utils.make_export_strategy(
        serving_input_fn, exports_to_keep=5,
        default_output_alternative_key=None)

    train_input_fn = get_transformed_reader_input_fn(
        transformed_metadata, args.train_data_paths, args.batch_size,
        tf.contrib.learn.ModeKeys.TRAIN)

    eval_input_fn = get_transformed_reader_input_fn(
        transformed_metadata, args.eval_data_paths, args.batch_size,
        tf.contrib.learn.ModeKeys.EVAL)

    print('got here')
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_steps=args.train_steps,
        #eval_steps=args.eval_steps,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        export_strategies=export_strategy,
        min_eval_frequency=500)

  # Return a function to create an Experiment.
  return get_experiment


def main(argv=None):
  """Run a Tensorflow model on the Reddit dataset."""
  env = json.loads(os.environ.get('TF_CONFIG', '{}'))
  # First find out if there's a task value on the environment variable.
  # If there is none or it is empty define a default one.
  task_data = env.get('task') or {'type': 'master', 'index': 0}
  argv = sys.argv if argv is None else argv
  args = create_parser().parse_args(args=argv[1:])

  trial = task_data.get('trial')
  if trial is not None:
    output_dir = os.path.join(args.output_path, trial)
  else:
    output_dir = args.output_path

  learn_runner.run(experiment_fn=get_experiment_fn(args),
                   output_dir=output_dir)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main()
