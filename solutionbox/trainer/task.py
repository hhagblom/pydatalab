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
import os
import re
import sys
import multiprocessing
import json


import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.lib.io import file_io

from tensorflow_transform.saved import input_fn_maker


SCHEMA_FILE = 'schema.json'
FEATURES_FILE = 'features.json'
NUMERICAL_ANALYSIS_FILE = 'stats.json'
VOCAB_ANALYSIS_FILE = 'vocab_%s.csv'

NUMERIC_TRANSFORMS = ['identity', 'scale']
CATEGORICAL_TRANSFORMS = ['one_hot', 'embedding']
TEXT_TRANSFORMS = ['bag_of_words', 'tfidf']

KEY_TRANSFORM = 'key'
TARGET_TRANSFORM = 'target'

TRANSFORMED_METADATA = 'transformed_metadata'
RAW_METADATA = 'raw_metadata'

def is_linear_model(model_type):
  return model_type.startswith('linear_')


def is_dnn_model(model_type):
  return model_type.startswith('dnn_')


def is_regression_model(model_type):
  return model_type.endswith('_regression')


def is_classification_model(model_type):
  return model_type.endswith('_classification')

def build_feature_columns(features, model_type):
  feature_columns = []
  for name, transform in six.iteritems(features):
    transform_name = transform['transform']

    if transform_name in NUMERIC_TRANSFORMS:
      feature_columns.append(tf.contrib.layers.real_valued_column(
          name,
          dimension=1))
    elif transform_name == 'one_hot':
      pass
    elif transform_name == 'embedding':
      pass
    elif transform_name == 'bag_of_words':
      pass
    elif transform_name == 'tfidf':
      pass
    else:
      raise ValueError('Unknown transfrom %s' % transform_name)


      # Supported transforms:
      # for DNN
      # 1) string -> make int -> embedding (embedding)
      # 2) string -> make int -> one_hot (one_hot, default)
      # for linear
      # 1) string -> sparse_column_with_hash_bucket (embedding)
      # 2) string -> make int -> sparse_column_with_integerized_feature (one_hot, default)
      # It is unfortunate that tf.layers has different feature transforms if the
      # model is linear or DNN. This pacakge should not expose to the user that
      # we are using tf.layers. It is crazy that DNN models support more feature
      # types (like string -> hash sparse column -> embedding)
  for name in train_config['categorical_columns']:
    if name != target_name and name != key_name:
      transform_config = train_config['transforms'].get(name, {})
      transform_name = transform_config.get('transform', None)

      if is_dnn_model(args.model_type):
        if transform_name == 'embedding':
          sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
              name,
              bucket_size=train_config['vocab_stats'][name]['n_classes'])
          learn_feature = tf.contrib.layers.embedding_column(
              sparse,
              dimension=transform_config['embedding_dim'])
        elif transform_name == 'one_hot' or transform_name is None:
          sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
              name,
              bucket_size=train_config['vocab_stats'][name]['n_classes'])
          learn_feature = tf.contrib.layers.one_hot_column(sparse)
        else:
          raise ValueError(('Unknown transform name. Only \'embedding\' '
                            'and \'one_hot\' transforms are supported. Got %s')
                           % transform_name)
      elif is_linear_model(args.model_type):
        if transform_name == 'one_hot' or transform_name is None:
          learn_feature = tf.contrib.layers.sparse_column_with_integerized_feature(
              name,
              bucket_size=train_config['vocab_stats'][name]['n_classes'])
        elif transform_name == 'embedding':
          learn_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
              name,
              hash_bucket_size=transform_config['embedding_dim'])
        else:
          raise ValueError(('Unknown transform name. Only \'embedding\' '
                            'and \'one_hot\' transforms are supported. Got %s')
                           % transform_name)

      # Save the feature
      feature_columns.append(learn_feature)
  return feature_columns


def get_estimator(output_dir, features, args, target_vocab_size):
  # Check layers used for dnn models.
  if is_dnn_model(args.model_type) and not args.layer_sizes:
    raise ValueError('--layer-size* must be used with DNN models')
  if is_linear_model(args.model_type) and args.layer_sizes:
    raise ValueError('--layer-size* cannot be used with linear models')

  # Build tf.learn features
  feature_columns = build_feature_columns(features)

  # Set how often to run checkpointing in terms of time.
  config = tf.contrib.learn.RunConfig(
      save_checkpoints_secs=args.save_checkpoints_secs)

  train_dir = os.path.join(output_dir, 'train')
  if args.model_type == 'dnn_regression':
    estimator = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=args.layer_sizes,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'linear_regression':
    estimator = tf.contrib.learn.LinearRegressor(
        feature_columns=feature_columns,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'dnn_classification':
    estimator = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=args.layer_sizes,
        n_classes=target_vocab_size,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  elif args.model_type == 'linear_classification':
    estimator = tf.contrib.learn.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=target_vocab_size,
        config=config,
        model_dir=train_dir,
        optimizer=tf.train.AdamOptimizer(
            args.learning_rate, epsilon=args.epsilon))
  else:
    raise ValueError('bad --model-type value')

  return estimator


def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))

def get_experiment_fn(args):
  """Builds the experiment function for learn_runner.run.

  Args:
    args: the command line args

  Returns:
    A function that returns a tf.learn experiment object.
  """

  def get_experiment(output_dir):
    # Merge schema, input features, and transforms.
    schema_file_path = os.path.join(args.analysis_output_dir, SCHEMA_FILE)
    if not file_io.file_exists(schema_file_path):
      raise ValueError('File not found: %s' % schema_file_path)
    schema = json.loads(file_io.read_file_to_string(schema_file_path))

    features_file_path = os.path.join(args.analysis_output_dir, FEATURES_FILE)
    if not file_io.file_exists(features_file_path):
      raise ValueError('File not found: %s' % features_file_path)
    features = json.loads(file_io.read_file_to_string(features_file_path))

    
    # Get the model to train.
    estimator = get_estimator(output_dir, features, args)

    
    # Make list of files to save with the trained model.
    additional_assets = {FEATURES_FILE: features_file_path,
                         SCHEMA_FILE: schema_file_path}

    #export_strategy_target = util.make_export_strategy(
    #    train_config=train_config,
    #    args=args,
    #    keep_target=True,
    #    assets_extra=additional_assets)
    #export_strategy_notarget = util.make_export_strategy(
    #    train_config=train_config,
    #    args=args,
    #    keep_target=False,
    #    assets_extra=additional_assets)

    target_column_name = None
    for name, transform in six.iteritems(features):
      if trnasform['transform'] == 'target':
        target_column_name = name
        break
    if not target_column_name:
      raise ValueError('Target transform missing') 

    transformed_metadata = metadata_io.read_metadata(
        os.path.join(args.analysis_output_dir, TRANSFORMED_METADATA))
    input_reader_for_train = input_fn_maker.build_training_input_fn(
        metadata=transformed_metadata,
        file_pattern=args.train_data_paths,
        training_batch_size=args.train_batch_size,
        reader=gzip_reader_fn,
        label_keys=[target_column_name],
        feature_keys=None,  # extract all features
        key_feature_name=None,  # None as we take care of the key column.
        reader_num_threads=multiprocessing.cpu_count(),
        queue_capacity=args.train_batch_size * multiprocessing.cpu_count() + 10,
        randomize_input=True,
        num_epochs=args.num_epochs,
    )
    input_reader_for_eval = input_fn_maker.build_training_input_fn(
        metadata=transformed_metadata,
        file_pattern=args.eval_data_paths,
        training_batch_size=args.eval_batch_size,
        reader=gzip_reader_fn,
        label_keys=[target_column_name],
        feature_keys=None,  # extract all features
        key_feature_name=None,  # None as we take care of the key column.
        reader_num_threads=multiprocessing.cpu_count(),
        queue_capacity=args.train_batch_size * multiprocessing.cpu_count() + 10,
        randomize_input=False,
        num_epochs=1,
    )

    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=input_reader_for_train,
        eval_input_fn=input_reader_for_eval,
        train_steps=args.max_steps,
        #export_strategies=[export_strategy_target, export_strategy_notarget],
        min_eval_frequency=args.min_eval_frequency,
        eval_steps=None,
    )

  # Return a function to create an Experiment.
  return get_experiment


def parse_arguments(argv):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser(
      description=('Train a regression or classification model. Note that if '
                   'using a DNN model, --layer-size1=NUM, --layer-size2=NUM, '
                   'should be used. '))

  # I/O file parameters
  parser.add_argument('--train-data-paths', type=str, action='append',
                      required=True)
  parser.add_argument('--eval-data-paths', type=str, action='append',
                      required=True)
  parser.add_argument('--job-dir', type=str, required=True)
  parser.add_argument('--analysis-output-dir',
                      type=str,
                      required=True,
                      help=('Output folder of analysis. Should contain the'
                            ' schema file, and numerical stats and vocab files.'
                            ' Path must be on GCS if running'
                            ' cloud training.'))
  #parser.add_argument('--transforms-file',
  #                    type=str,
  #                    required=True,
  #                    help=('File describing the the transforms to apply on '
  #                          'each column'))

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
                      required=True)
  parser.add_argument('--top-n',
                      type=int,
                      default=1,
                      help=('For classification problems, the output graph '
                            'will contain the labels and scores for the top '
                            'n classes.'))
  # Training input parameters
  parser.add_argument('--max-steps', type=int, default=5000,
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