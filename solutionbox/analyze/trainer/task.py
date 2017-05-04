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
import pandas as pd
import re
import six
import sys
import multiprocessing
import json


import tensorflow as tf
from tensorflow.python.framework import dtypes

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.lib.io import file_io

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io

from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import dataset_schema

# Files
SCHEMA_FILE = 'schema.json'
FEATURES_FILE = 'features.json'
STATS_FILE = 'stats.json'
VOCAB_ANALYSIS_FILE = 'vocab_%s.csv'

TRANSFORMED_METADATA_DIR = 'transformed_metadata'
RAW_METADATA_DIR = 'raw_metadata'
TRANSFORM_FN_DIR = 'transform_fn'

# Individual transforms
IDENTITY_TRANSFORM = 'identity'
SCALE_TRANSFORM = 'scale'
ONE_HOT_TRANSFORM = 'one_hot'
EMBEDDING_TRANSFROM = 'embedding'
BOW_TRANSFORM = 'bag_of_words'
TFIDF_TRANSFORM = 'tfidf'
KEY_TRANSFORM = 'key'
TARGET_TRANSFORM = 'target'

# Transform collections
NUMERIC_TRANSFORMS = [IDENTITY_TRANSFORM, SCALE_TRANSFORM]
CATEGORICAL_TRANSFORMS = [ONE_HOT_TRANSFORM, EMBEDDING_TRANSFROM]
TEXT_TRANSFORMS = [BOW_TRANSFORM, TFIDF_TRANSFORM]

def parse_arguments(argv):
  """Parse the command line arguments."""
  parser = argparse.ArgumentParser(
      description=('Train a regression or classification model. Note that if '
                   'using a DNN model, --layer-size1=NUM, --layer-size2=NUM, '
                   'should be used. '))

  # I/O file parameters
  parser.add_argument('--train-data-paths', type=str,
                      required=True)
  parser.add_argument('--eval-data-paths', type=str,
                      required=True)
  parser.add_argument('--job-dir', type=str, required=True)
  parser.add_argument('--analysis-output-dir',
                      type=str,
                      required=True,
                      help=('Output folder of analysis. Should contain the'
                            ' schema, stats, and vocab files.'
                            ' Path must be on GCS if running'
                            ' cloud training.'))
  parser.add_argument('--run-transforms',
                      action='store_true',
                      default=False,
                      help=('If used, input data is raw csv that needs '
                            'transformation.'))  

  # HP parameters
  parser.add_argument('--learning-rate', type=float, default=0.01,
                      help='tf.train.AdamOptimizer learning rate')
  parser.add_argument('--epsilon', type=float, default=0.0005,
                      help='tf.train.AdamOptimizer epsilon')

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


def is_linear_model(model_type):
  return model_type.startswith('linear_')


def is_dnn_model(model_type):
  return model_type.startswith('dnn_')


def is_regression_model(model_type):
  return model_type.endswith('_regression')


def is_classification_model(model_type):
  return model_type.endswith('_classification')

def build_feature_columns(features, stats, model_type):
  feature_columns = []
  _is_dnn_model = is_dnn_model(model_type)

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
  for name, transform in six.iteritems(features):
    transform_name = transform['transform']

    if transform_name in NUMERIC_TRANSFORMS:
      new_feature = tf.contrib.layers.real_valued_column(name, dimension=1)
    elif transform_name == ONE_HOT_TRANSFORM:
      sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
          name,
          bucket_size=stats['column_stats'][name]['vocab_size'])
      if _is_dnn_model:
        new_feature = tf.contrib.layers.one_hot_column(sparse)
      else:
        new_feature = sparse
    elif transform_name == EMBEDDING_TRANSFROM:
      if _is_dnn_model:
        sparse = tf.contrib.layers.sparse_column_with_integerized_feature(
            name,
            bucket_size=stats['column_stats'][name]['vocab_size'])
        new_feature = tf.contrib.layers.embedding_column(
            sparse,
            dimension=transform['embedding_dim'])
      else:
        new_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
            name,
            hash_bucket_size=transform['embedding_dim'],
            dtype=dtypes.int64)
    elif transform_name in TEXT_TRANSFORMS:
      sparse_ids = tf.contrib.layers.sparse_column_with_integerized_feature(
          name + '_ids',
          bucket_size=stats['column_stats'][name]['vocab_size'])
      sparse_weights =  tf.contrib.layers.weighted_sparse_column(
          sparse_id_column=sparse_ids, 
          weight_column_name=name + '_weights',
          dtype=dtypes.float32)
      if _is_dnn_model:
        new_feature = sparse_weights # TODO(brandondutra): is this correct? or need one-hot?
      else:
        new_feature = sparse_weights
    elif transform_name == TARGET_TRANSFORM or transform_name == KEY_TRANSFORM:
      continue
    else:
      raise ValueError('Unknown transfrom %s' % transform_name)

    feature_columns.append(new_feature)

  return feature_columns


def build_csv_transforming_training_input_fn(raw_metadata,
                                             transform_savedmodel_dir,
                                             raw_data_file_pattern,
                                             training_batch_size,
                                             raw_keys,
                                             transformed_label_keys,
                                             convert_scalars_to_vectors=True,
                                             num_epochs=None,
                                             randomize_input=False,
                                             min_after_dequeue=1,
                                             reader_num_threads=1):
  """Creates training input_fn that reads raw csv data and applies transforms.

  Args:
    raw_metadata: a `DatasetMetadata` object describing the raw data.
    transform_savedmodel_dir: a SavedModel directory produced by tf.Transform
      embodying a transformation function to be applied to incoming raw data.
    raw_data_file_pattern: List of files or pattern of file paths containing
        `Example` records. See `tf.gfile.Glob` for pattern rules.
    training_batch_size: An int or scalar `Tensor` specifying the batch size to
      use.
    raw_keys: List of string keys giving the order in the csv file.
    transformed_label_keys
    convert_scalars_to_vectors: Boolean specifying whether this input_fn should
      convert scalars into 1-d vectors.  This is necessary if the inputs will be
      used with `FeatureColumn`s as `FeatureColumn`s cannot accept scalar
      inputs. Default: True.

    num_epochs
    randomize_input
    min_after_dequeue
    reader_num_threads
    queue_capacity

  Returns:
    An input_fn suitable for training that reads raw csv training data and
    applies transforms.

  """

  if not raw_keys:
    raise ValueError("raw_keys must be set.")

  column_schemas = raw_metadata.schema.column_schemas

  # Check for errors.
  for k in raw_keys:
    if k not in column_schemas:
      raise ValueError("Key %s does not exist in the schema" % k)
    if not isinstance(column_schemas[k].representation,
                      dataset_schema.FixedColumnRepresentation):
      raise ValueError(("CSV files can only support tensors of fixed size"
                        "which %s is not.") % k)
    shape = column_schemas[k].tf_shape().as_list()
    if shape and shape != [1]:
      # Column is not a scalar-like value. shape == [] or [1] is ok.
      raise ValueError(("CSV files can only support features that are scalars "
                        "having shape []. %s has shape %s")
                       % (k, shape))


  def raw_training_input_fn():
    """Training input function that reads raw data and applies transforms."""

    if isinstance(raw_data_file_pattern, six.string_types):
      filepath_list = [raw_data_file_pattern]

    files = []
    for path in filepath_list:
      files.extend(file_io.get_matching_files(path))

    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=randomize_input)

    csv_id, csv_lines = tf.TextLineReader().read_up_to(filename_queue, training_batch_size)

    queue_capacity = (reader_num_threads + 3) * training_batch_size + min_after_dequeue
    if randomize_input:
      batch_csv_id, batch_csv_lines = tf.train.shuffle_batch(
          tensors=[csv_id, csv_lines],
          batch_size=training_batch_size,
          capacity=queue_capacity,
          min_after_dequeue=min_after_dequeue,
          enqueue_many=True,
          num_threads=reader_num_threads)

    else:
      batch_csv_id, batch_csv_lines = tf.train.batch(
          tensors=[csv_id, csv_lines],
          batch_size=batch_size,
          capacity=queue_capacity,
          enqueue_many=True,
          num_threads=reader_num_threads)

    record_defaults = []
    for k in raw_keys:
      if column_schemas[k].representation.default_value:
        value = tf.constant([column_schemas[k].representation.default_value],
                            dtype=column_schemas[k].domain.dtype)
      else:
        value = tf.constant([], dtype=column_schemas[k].domain.dtype)
      record_defaults.append(value)


    parsed_tensors  = tf.decode_csv(batch_csv_lines, record_defaults, name='csv_to_tensors')

    raw_data = {k: v for k, v in zip(raw_keys, parsed_tensors)}
    print(raw_data)

    transformed_data = saved_transform_io.apply_saved_transform(
        transform_savedmodel_dir, raw_data)

    transformed_features = {
        k: v for k, v in six.iteritems(transformed_data)
        if k not in transformed_label_keys}
    transformed_labels = {
        k: v for k, v in six.iteritems(transformed_data)
        if k in transformed_label_keys}

    if convert_scalars_to_vectors:
      transformed_features = input_fn_maker._convert_scalars_to_vectors(transformed_features)
      transformed_labels = input_fn_maker._convert_scalars_to_vectors(transformed_labels)

    # TODO(b/35264116): remove this when all estimators accept label dict
    if len(transformed_labels) == 1:
      (_, transformed_labels), = transformed_labels.items()
    return transformed_features, transformed_labels

  return raw_training_input_fn



def get_estimator(args, output_dir, features, stats, target_vocab_size):
  # Check layers used for dnn models.
  if is_dnn_model(args.model_type) and not args.layer_sizes:
    raise ValueError('--layer-size* must be used with DNN models')
  if is_linear_model(args.model_type) and args.layer_sizes:
    raise ValueError('--layer-size* cannot be used with linear models')

  # Build tf.learn features
  feature_columns = build_feature_columns(features, stats, args.model_type)

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

def read_vocab(args, column_name):
  """Reads a vocab file if it exists.

  Args:
    args: command line flags
    column_name: name of column to that has a vocab file.

  Returns:
    List of vocab words or [] if the vocab file is not found.
  """
  vocab_path = os.path.join(args.analysis_output_dir, VOCAB_ANALYSIS_FILE % column_name)

  if not file_io.file_exists(vocab_path):
    return []

  vocab_str = file_io.read_file_to_string(vocab_path)
  vocab = pd.read_csv(six.StringIO(vocab_str),
                      header=None,
                      names=['token', 'count'])
  return vocab['token'].tolist()


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
    schema = json.loads(file_io.read_file_to_string(schema_file_path).decode())

    features_file_path = os.path.join(args.analysis_output_dir, FEATURES_FILE)
    if not file_io.file_exists(features_file_path):
      raise ValueError('File not found: %s' % features_file_path)
    features = json.loads(file_io.read_file_to_string(features_file_path).decode())

    stats_file_path = os.path.join(args.analysis_output_dir, STATS_FILE)
    if not file_io.file_exists(stats_file_path):
      raise ValueError('File not found: %s' % stats_file_path)
    stats = json.loads(file_io.read_file_to_string(stats_file_path).decode())

    target_column_name = None
    key_column_name = None
    header_names = []
    for name, transform in six.iteritems(features):
      header_names.append(name)
      if transform['transform'] == TARGET_TRANSFORM:
        target_column_name = name
      elif transform['transform'] == KEY_TRANSFORM:
        key_column_name = name
    if not target_column_name or not key_column_name:
      raise ValueError('target or key transform missing from features file.') 
    
    # Get the model to train.
    target_vocab = read_vocab(args, target_column_name)
    estimator = get_estimator(args, output_dir, features, stats, len(target_vocab))

    
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

    
    transformed_metadata = metadata_io.read_metadata(
        os.path.join(args.analysis_output_dir, TRANSFORMED_METADATA_DIR))
    if args.run_transforms:
      raw_metadata = metadata_io.read_metadata(
        os.path.join(args.analysis_output_dir, RAW_METADATA_DIR))

      input_reader_for_train = build_csv_transforming_training_input_fn(
          raw_metadata=raw_metadata,
          transform_savedmodel_dir=os.path.join(args.analysis_output_dir, 'transform_fn'),
          raw_data_file_pattern=args.train_data_paths,
          training_batch_size=args.train_batch_size,
          raw_keys=header_names,
          transformed_label_keys=[target_column_name],
          convert_scalars_to_vectors=True,
          num_epochs=args.num_epochs,
          randomize_input=True,
          min_after_dequeue=10,
          reader_num_threads=multiprocessing.cpu_count()
      )
      input_reader_for_eval = build_csv_transforming_training_input_fn(
          raw_metadata=raw_metadata,
          transform_savedmodel_dir=os.path.join(args.analysis_output_dir, 'transform_fn'),
          raw_data_file_pattern=args.eval_data_paths,
          training_batch_size=args.eval_batch_size,
          raw_keys=header_names,
          transformed_label_keys=[target_column_name],
          convert_scalars_to_vectors=True,
          num_epochs=1,
          randomize_input=False,
          reader_num_threads=multiprocessing.cpu_count()
      )      
    else:
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





def main(argv=None):
  """Run a Tensorflow model on the Iris dataset."""
  args = parse_arguments(sys.argv if argv is None else argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  learn_runner.run(
      experiment_fn=get_experiment_fn(args),
      output_dir=args.job_dir)


if __name__ == '__main__':
  main()
