# Copyright 2017 Google Inc. All Rights Reserved.
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

# Flake8 cannot disable a warning for the file. Flake8 does not like beam code
# and reports many 'W503 line break before binary operator' errors. So turn off
# flake8 for this file.
# flake8: noqa
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import json
import logging
import os
import random
import sys
import apache_beam as beam
from apache_beam.metrics import Metrics
from PIL import Image
import six
from tensorflow.python.lib.io import file_io
from tensorflow_transform import coders
from tensorflow_transform.beam import impl as tft
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import metadata_io


img_error_count = Metrics.counter('main', 'ImgErrorCount')

# Files
SCHEMA_FILE = 'schema.json'
FEATURES_FILE = 'features.json'

TRANSFORMED_METADATA_DIR = 'transformed_metadata'
RAW_METADATA_DIR = 'raw_metadata'
TRANSFORM_FN_DIR = 'transform_fn'

# Individual transforms
TARGET_TRANSFORM = 'target'
IMAGE_URL_TO_VEC_TRANSFORM = 'img_url_to_vec'


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
      '--project-id',
      help='The project to which the job will be submitted.')
  parser.add_argument(
      '--cloud',
      action='store_true',
      help='Run preprocessing on the cloud.')
  parser.add_argument(
      '--job-name',
      type=str,
      help='Unique job name if running on the cloud.')

  parser.add_argument(
      '--csv-file-pattern',
      required=True,
      help='CSV data to encode as tf.example.')

  parser.add_argument(
      '--analyze-output-dir',
      required=True,
      help='The output folder of analyze')
  parser.add_argument(
      '--output-filename-prefix',
      required=True,
      type=str)
  parser.add_argument(
      '--output-dir',
      default=None,
      required=True,
      help=('Google Cloud Storage or Local directory in which '
            'to place outputs.'))

  feature_parser = parser.add_mutually_exclusive_group(required=False)
  feature_parser.add_argument('--target', dest='target', action='store_true')
  feature_parser.add_argument('--no-target', dest='target', action='store_false')
  parser.set_defaults(feature=True)

  args, _ = parser.parse_known_args(args=argv[1:])

  if args.cloud and not args.project_id:
    raise ValueError('--project-id is needed for --cloud')

  if not args.job_name:
    args.job_name = ('dataflow-job-{}'.format(
        datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
  return args



def preprocess(pipeline, args):
  input_metadata = metadata_io.read_metadata(
      os.path.join(args.analyze_output_dir, RAW_METADATA_DIR))

  schema = json.loads(file_io.read_file_to_string(
      os.path.join(args.analyze_output_dir, SCHEMA_FILE)).decode())
  features = json.loads(file_io.read_file_to_string(
      os.path.join(args.analyze_output_dir, FEATURES_FILE)).decode())

  column_names = [col['name'] for col in schema]
  if not args.target:
    target_name = None
    for name, transform in six.iteritems(features):
      if transform['transform'] == TARGET_TRANSFORM:
        target_name = name
        break
    column_names.remove(target_name)
    del input_metadata.schema.column_schemas[target_name]

  csv_coder = coders.CsvCoder(column_names, input_metadata.schema, delimiter=',')
  tfex_coder = coders.ExampleProtoCoder(input_metadata.schema)
  raw_data = (
      pipeline
      | 'ReadCsvData' >> beam.io.ReadFromText(args.csv_file_pattern)
      | 'ParseCsvData' >> beam.Map(csv_coder.decode)
      | 'EncodeTFExample' >> beam.Map(tfex_coder.encode)
      | 'WriteTFExampleTFRecord' >> beam.io.WriteToTFRecord(
             os.path.join(args.output_dir, args.output_filename_prefix),
             file_name_suffix='.tfrecord.gz'))

def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)
  if args.cloud:
    pipeline_name = 'DataflowRunner'
    options = {
        'job_name': args.job_name,
        'temp_location':
            os.path.join(args.output_dir, 'tmp'),
        'project':
            args.project_id,
    }

    pipeline_options = beam.pipeline.PipelineOptions(flags=[], **options)
  else:
    pipeline_name = 'DirectRunner'
    pipeline_options = None

  temp_dir = os.path.join(args.output_dir, 'tmp')
  with beam.Pipeline(pipeline_name, options=pipeline_options) as p:
    with tft.Context(temp_dir=temp_dir):
      preprocess(
          pipeline=p,
          args=args)


if __name__ == '__main__':
  main()
