# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Criteo Classification Sample Preprocessing Runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import subprocess
import sys

#import criteo
#import path_constants

TRANSFORM_FN_DIR = 'transform_fn'
RAW_METADATA_DIR = 'raw_metadata'
TRANSFORMED_METADATA_DIR = 'transformed_metadata'
TRANSFORMED_TRAIN_DATA_FILE_PREFIX = 'features_train'
TRANSFORMED_EVAL_DATA_FILE_PREFIX = 'features_eval'
TRANSFORMED_PREDICT_DATA_FILE_PREFIX = 'features_predict'
TRAIN_RESULTS_FILE = 'train_results'
DEPLOY_SAVED_MODEL_DIR = 'saved_model'
MODEL_EVALUATIONS_FILE = 'model_evaluations'
BATCH_PREDICTION_RESULTS_FILE = 'batch_prediction_results'


import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft

from tensorflow_transform import coders
from tensorflow_transform import version as tft_version
from tensorflow_transform.beam import impl as tft_impl
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema



def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments including program name.
  Returns:
    The parsed arguments as returned by argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser(
      description='Runs Preprocessing on the Criteo model data.')

  parser.add_argument(
      '--training_data',
      default='train_csv_data.csv',
      help='Data to analyze and encode as training features.')
  parser.add_argument(
      '--eval_data',
      default='eval_csv_data.csv',
      help='Data to encode as evaluation features.')
  parser.add_argument(
      '--predict_data', help='Data to encode as prediction features.')
  parser.add_argument(
      '--output_dir',
      default='tfpreout',
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))
  args, _ = parser.parse_known_args(args=argv[1:])


  return args

def make_input_schema(mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Input schema definition.

  Args:
    mode: tf.contrib.learn.ModeKeys specifying if the schema is being used for
      train/eval or prediction.
  Returns:
    A `Schema` object.
  """
  # put default values here: default_value='
  result = {}
  result['key'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
  if mode == tf.contrib.learn.ModeKeys.TRAIN:
    result['target'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
  result['num1'] = tf.FixedLenFeature(shape=[], dtype=tf.float64)
  result['num2'] = tf.FixedLenFeature(shape=[], dtype=tf.float64)
  result['num3'] = tf.FixedLenFeature(shape=[], dtype=tf.float64)
  result['str1'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
  result['str2'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
  result['str3'] = tf.FixedLenFeature(shape=[], dtype=tf.string)

  return dataset_schema.from_feature_spec(result)

def make_tsv_coder(schema, mode=tf.contrib.learn.ModeKeys.TRAIN):
  """Produces a CsvCoder (with tab as the delimiter) from a data schema.

  Args:
    schema: A tf.Transform `Schema` object.
    mode: tf.contrib.learn.ModeKeys specifying if the source is being used for
      train/eval or prediction.

  Returns:
    A tf.Transform CsvCoder.
  """
  column_names = ['key'] 
  if mode == tf.contrib.learn.ModeKeys.TRAIN:
    column_names.append('target')
  column_names.append('num1')
  column_names.append('num2')
  column_names.append('num3')
  column_names.append('str1')
  column_names.append('str2')
  column_names.append('str3')

  return coders.CsvCoder(column_names, schema, delimiter=',')



def make_preprocessing_fn():
  def preprocessing_fn(inputs):
    """
    Args:
      inputs: dictionary of input `tensorflow_transform.Column`.
    Returns:
      A dictionary of `tensorflow_transform.Column` representing the transformed
          columns.
    """
    
    result = {'key': inputs['key']}
    result['target'] = tft.string_to_int(inputs['target'])
    result['num1'] = tft.scale_to_0_1(inputs['num1'])
    result['num2'] = tft.scale_to_0_1(inputs['num2'])
    result['num3'] = tft.scale_to_0_1(inputs['num3'])
    result['str1'] = tft.string_to_int(inputs['str1'])
    result['str2'] = tft.string_to_int(inputs['str2'])
    result['str3'] = tft.string_to_int(inputs['str3'])
    return result

  return preprocessing_fn

def preprocess(pipeline, training_data, eval_data, predict_data, output_dir):
  # 1) The schema can be either defined in-memory or read from a configuration
  #    file, in this case we are creating the schema in-memory.
  input_schema = make_input_schema()
  print('input_schema', input_schema)

  # 2) Configure the coder to map the source file column names to a dictionary
  #    of key -> tensor_proto with the appropiate type derived from the
  #    input_schema.
  coder = make_tsv_coder(input_schema)
  print('coder', coder)

  # 3) Read from text using the coder.
  train_data = (
      pipeline
      | 'ReadTrainingData' >> beam.io.ReadFromText(training_data)
      | 'ParseTrainingCsv' >> beam.Map(coder.decode))

  evaluate_data = (
      pipeline
      | 'ReadEvalData' >> beam.io.ReadFromText(eval_data)
      | 'ParseEvalCsv' >> beam.Map(coder.decode))

  input_metadata = dataset_metadata.DatasetMetadata(schema=input_schema)
  _ = (input_metadata
       | 'WriteInputMetadata' >> tft_beam_io.WriteMetadata(
           os.path.join(output_dir, RAW_METADATA_DIR),
           pipeline=pipeline))

  preprocessing_fn = make_preprocessing_fn()
  (train_dataset, train_metadata), transform_fn = (
      (train_data, input_metadata)
      | 'AnalyzeAndTransform' >> tft_impl.AnalyzeAndTransformDataset(
          preprocessing_fn))

  # WriteTransformFn writes transform_fn and metadata to fixed subdirectories
  # of output_dir, which are given by path_constants.TRANSFORM_FN_DIR and
  # path_constants.TRANSFORMED_METADATA_DIR.
  _ = (transform_fn | 'WriteTransformFn' >> tft_beam_io.WriteTransformFn(output_dir))

  # TODO(b/34231369) Remember to eventually also save the statistics.

  (evaluate_dataset, evaluate_metadata) = (
      ((evaluate_data, input_metadata), transform_fn)
      | 'TransformEval' >> tft_impl.TransformDataset())

  train_coder = coders.ExampleProtoCoder(train_metadata.schema)
  _ = (train_dataset
       | 'SerializeTrainExamples' >> beam.Map(train_coder.encode)
       | 'WriteTraining'
       >> beam.io.WriteToTFRecord(
           os.path.join(output_dir,
                        TRANSFORMED_TRAIN_DATA_FILE_PREFIX),
           file_name_suffix='.tfrecord.gz'))

  evaluate_coder = coders.ExampleProtoCoder(evaluate_metadata.schema)
  _ = (evaluate_dataset
       | 'SerializeEvalExamples' >> beam.Map(evaluate_coder.encode)
       | 'WriteEval'
       >> beam.io.WriteToTFRecord(
           os.path.join(output_dir,
                        TRANSFORMED_EVAL_DATA_FILE_PREFIX),
           file_name_suffix='.tfrecord.gz'))

  if predict_data:
    predict_mode = tf.contrib.learn.ModeKeys.INFER
    predict_schema = criteo.make_input_schema(mode=predict_mode)
    tsv_coder = criteo.make_tsv_coder(predict_schema, mode=predict_mode)
    predict_coder = coders.ExampleProtoCoder(predict_schema)
    serialized_examples = (
        pipeline
        | 'ReadPredictData' >> beam.io.ReadFromText(predict_data)
        | 'ParsePredictCsv' >> beam.Map(tsv_coder.decode)
        # TODO(b/35194257) Obviate the need for this explicit serialization.
        | 'EncodePredictData' >> beam.Map(predict_coder.encode))
    _ = (serialized_examples
         | 'WritePredictDataAsTFRecord' >> beam.io.WriteToTFRecord(
             os.path.join(output_dir,
                          TRANSFORMED_PREDICT_DATA_FILE_PREFIX),
             file_name_suffix='.tfrecord.gz'))
    _ = (serialized_examples
         | 'EncodePredictAsB64Json' >> beam.Map(_encode_as_b64_json)
         | 'WritePredictDataAsText' >> beam.io.WriteToText(
             os.path.join(output_dir,
                          TRANSFORMED_PREDICT_DATA_FILE_PREFIX),
             file_name_suffix='.txt'))


def _encode_as_b64_json(serialized_example):
  import base64  # pylint: disable=g-import-not-at-top
  import json  # pylint: disable=g-import-not-at-top
  return json.dumps({'b64': base64.b64encode(serialized_example)})


def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)

  pipeline_name = 'DirectRunner'
  pipeline_options = None

  temp_dir = os.path.join(args.output_dir, 'tmp')
  with beam.Pipeline(pipeline_name, options=pipeline_options) as p:
    with tft_impl.Context(temp_dir=temp_dir):
      preprocess(
          pipeline=p,
          training_data=args.training_data,
          eval_data=args.eval_data,
          predict_data=args.predict_data,
          output_dir=args.output_dir)


if __name__ == '__main__':
  main()

