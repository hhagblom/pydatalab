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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import random
import subprocess
import sys

import json
from tensorflow.python.lib.io import file_io

import apache_beam as beam
import tensorflow as tf

from tensorflow_transform import coders
from tensorflow_transform.beam import impl as tft
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import dataset_metadata


from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform import impl_helper
from tensorflow_transform.tf_metadata import metadata_io



TRANSFORMED_METADATA_DIR = 'transformed_metadata'
RAW_METADATA_DIR = 'raw_metadata'
TRANSFORM_FN_DIR = 'transform_fn'

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
      '--project-id', help='The project to which the job will be submitted.')
  parser.add_argument(
      '--cloud', action='store_true', help='Run preprocessing on the cloud.')

  parser.add_argument(
      '--csv-file-pattern',
      required=True,
      help='Data to encode as tf.example.')
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
  args, _ = parser.parse_known_args(args=argv[1:])

  if args.cloud and not args.project_id:
    raise ValueError('--project-id is needed for --cloud')

  return args


# TODO(b/33688220) should the transform functions take shuffle as an optional
# argument instead?
@beam.ptransform_fn
def _Shuffle(pcoll):  # pylint: disable=invalid-name
  return (pcoll
          | 'PairWithRandom' >> beam.Map(lambda x: (random.random(), x))
          | 'GroupByRandom' >> beam.GroupByKey()
          | 'DropRandom' >> beam.FlatMap(lambda (k, vs): vs))


def preprocess(pipeline, args):

  output_metadata = metadata_io.read_metadata(os.path.join(args.analyze_output_dir, TRANSFORMED_METADATA_DIR))
  input_metadata = metadata_io.read_metadata(os.path.join(args.analyze_output_dir, RAW_METADATA_DIR))

  schema = json.loads(file_io.read_file_to_string(os.path.join(args.analyze_output_dir, 'schema.json')))
  column_names = [col['name'] for col in schema]
  coder = coders.CsvCoder(column_names, input_metadata.schema, delimiter=',')

  raw_data = (
      pipeline
      | 'ReadRawData' >> beam.io.ReadFromText(args.csv_file_pattern)
      | 'ParseCsvData' >> beam.Map(coder.decode))

  transform_fn = (
      pipeline 
      | 'ReadTransformFn' 
      >> tft_beam_io.ReadTransformFn(args.analyze_output_dir))
  # tft_beam_io.ReadTransformFn(os.path.join(args.analyze_output_dir, TRANSFORM_FN_DIR)))


  (transformed_data, transform_metadata) = (
      ((raw_data, input_metadata), transform_fn)
      | 'TransformEval' >> tft.TransformDataset())

  tfexample_coder = coders.ExampleProtoCoder(transform_metadata.schema)
  _ = (transformed_data
       | 'SerializeExamples' >> beam.Map(tfexample_coder.encode)
       #| 'ShuffleTraining' >> _Shuffle()  # pylint: disable=no-value-for-parameter
       | 'WriteExamples'
       >> beam.io.WriteToTFRecord(
           os.path.join(args.output_dir, args.output_filename_prefix),
           file_name_suffix='.tfrecord.gz'))


def main(argv=None):
  """Run Preprocessing as a Dataflow."""
  args = parse_arguments(sys.argv if argv is None else argv)
  if args.cloud:
    pipeline_name = 'DataflowRunner'
    options = {
        'job_name': ('cloud-ml-sample-criteo-preprocess-{}'.format(
            datetime.datetime.now().strftime('%Y%m%d%H%M%S'))),
        'temp_location':
            os.path.join(args.output_dir, 'tmp'),
        'project':
            args.project_id,
        # TODO(b/35727492): Remove this.
        'max_num_workers':
            1000,
        'setup_file':
            os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                'setup.py')),
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