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

# run via
# python -m trainer/task --train-data-paths train_csv_data.csv  --eval-data-paths eval_csv_data.csv  --job-dir TOUT --preprocess-output-dir x --transforms-file y --model-type dnn_classification --top-n 3


#from . import util
import tensorflow as tf

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.lib.io import file_io


def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))

def get_reader_input_fn(data_paths, batch_size, shuffle, num_epochs=None):
  """Builds input layer for training."""
  transformed_metadata = metadata_io.read_metadata(
        'tfpreout/transformed_metadata')

  print('tf metadata')
  print(transformed_metadata)
  print(dir(transformed_metadata))
  print(dir(transformed_metadata.schema))
  return input_fn_maker.build_training_input_fn(
      metadata=transformed_metadata,
      file_pattern=data_paths,
      training_batch_size=batch_size,
      label_keys=['target'],
      reader=gzip_reader_fn,
      key_feature_name='key',
      reader_num_threads=4,
      queue_capacity=batch_size * 2,
      randomize_input=shuffle,
      num_epochs=num_epochs)
  # def get_input_features():
  #   """Read the input features from the given data paths."""
  #   _, examples = util.read_examples(
  #       input_files=data_paths,
  #       batch_size=batch_size,
  #       shuffle=shuffle,
  #       num_epochs=num_epochs)
  #   features = util.parse_example_tensor(examples=examples,
  #                                        train_config=train_config,
  #                                        keep_target=True)

  #   target_name = train_config['target_column']
  #   target = features.pop(target_name)
  #   features, target = util.preprocess_input(
  #       features=features,
  #       target=target,
  #       train_config=train_config,
  #       preprocess_output_dir=preprocess_output_dir,
  #       model_type=model_type)

  #   return features, target

  # # Return a function to input the feaures into the model from a data path.
  # return get_input_features



def get_estimator(train_root, args):

  s1 = tf.contrib.layers.sparse_column_with_integerized_feature('str1', bucket_size=8+1) # bucket_size = vocab_size + unknown label
  s2 = tf.contrib.layers.sparse_column_with_integerized_feature('str2', bucket_size=7+1)
  s3 = tf.contrib.layers.sparse_column_with_integerized_feature('str3', bucket_size=7+1)

  feature_columns = [
      tf.contrib.layers.real_valued_column('num1', dimension=1),
      tf.contrib.layers.real_valued_column('num2', dimension=1),
      tf.contrib.layers.real_valued_column('num3', dimension=1),
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
        'tfpreout/transformed_metadata')
    raw_metadata = metadata_io.read_metadata('tfpreout/raw_metadata')
    serving_input_fn = (
        input_fn_maker.build_parsing_transforming_serving_input_fn(
            raw_metadata,
            'tfpreout/transform_fn',
            raw_label_keys=['target']))
    export_strategy = tf.contrib.learn.utils.make_export_strategy(
        serving_input_fn, exports_to_keep=5,
        default_output_alternative_key=None)



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
        export_strategies=[export_strategy],
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
                      default='../train_csv_data.csv')
  parser.add_argument('--eval-data-paths', type=str, action='append',
                      default='../eval_csv_data.csv')
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
