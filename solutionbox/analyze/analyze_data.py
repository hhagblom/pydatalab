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
import collections
import json
import os
import six
import sys


from tensorflow.python.lib.io import file_io

SCHEMA_FILE = 'schema.json'
NUMERICAL_ANALYSIS_FILE = 'stats.json'
VOCAB_ANALYSIS_FILE = 'vocab_%s.csv'


NUMERIC_TRANSFORMS = ['identity', 'scale']
CATEGORICAL_TRANSFORMS = ['one_hot', 'embedding']
TEXT_TRANSFORMS = ['bag_of_words']

 
NUMERIC_SCHEMA = ['integer', 'float']
STRING_SCHEMA  = ['string']
SUPPORTED_SCHEMA = NUMERIC_SCHEMA + STRING_SCHEMA

def parse_arguments(argv):
  """Parse command line arguments.

  Args:
    argv: list of command line arguments, includeing programe name.

  Returns:
    An argparse Namespace object.

  Raises:
    ValueError: for bad parameters
  """
  parser = argparse.ArgumentParser(
      description='Runs analysis on structured data.')
  parser.add_argument('--cloud',
                      action='store_true',
                      help='Analysis will use cloud services.')
  parser.add_argument('--output-dir',
                      type=str,
                      required=True,
                      help='GCS or local folder')

  # CSV inputs
  parser.add_argument('--csv-file-pattern',
                      type=str,
                      required=False,
                      help='Input CSV file names. May contain a file pattern')
  parser.add_argument('--csv-schema-file',
                      type=str,
                      required=False,
                      help=('BigQuery json schema file'))

  # If using bigquery table
  # TODO(brandondutra): maybe also support an sql input, so the table can be
  # ad-hoc.
  parser.add_argument('--bigquery-table',
                      type=str,
                      required=False,
                      help=('project:dataset.table_name'))

  parser.add_argument('--features-file',
                      type=str,
                      required=True,
                      help='File listing transforms to perform.')

  args = parser.parse_args(args=argv[1:])

  if args.cloud:
    if not args.output_dir.startswith('gs://'):
      raise ValueError('--output-dir must point to a location on GCS')
    if args.csv_file_pattern and not args.csv_file_pattern.startswith('gs://'):
      raise ValueError('--csv-file-pattern must point to a location on GCS')
    if args.csv_schema_file and not args.csv_schema_file.startswith('gs://'):
      raise ValueError('--csv-schema-file must point to a location on GCS')

  if not ( (args.bigquery_table and args.csv_file_pattern is None 
            and args.csv_schema_file is None)
          or (args.bigquery_table is None and args.csv_file_pattern 
            and args.csv_schema_file)):
    raise ValueError('either --csv-schema-file and --csv-file-pattern must both'
                     ' be set or just --bigquery-table is set')

  return args


def run_cloud_analysis(args, schema, features):
  pass


def run_local_analysis(args, schema, features):
  header = [column['name'] for column in schema]
  input_files = file_io.get_matching_files(args.csv_file_pattern)

  # initialize the results
  def _init_numerical_results():
    return {'min': float('inf'),
            'max': float('-inf'),
            'count': 0,
            'sum': 0.0}
  numerical_results = collections.defaultdict(_init_numerical_results)
  vocabs = collections.defaultdict(lambda: collections.defaultdict(int))


  transforms_to_do = get_transforms_per_input_column(schema, features)
  

  # for each file, update the numerical stats from that file, and update the set
  # of unique labels.
  for input_file in input_files:
    with file_io.FileIO(input_file, 'r') as f:
      for line in csv.reader(f):
        parsed_line = dict(zip(header, line))

        for col_name in header:
          if transforms_to_do[name] <= set(NUMERIC_TRANSFORMS):
            aaa

          elif transforms_to_do[name] <= set(CATEGORICAL_TRANSFORMS):

          elif  transforms_to_do[name] <= set(TEXT_TRANSFORMS):
            split_strings = parsed_line[col_name].split(' ')

            for one_label in split_strings:
              # Filter out empty strings
              if one_label:
                # add the label to the dict and increase its count.
                categorical_results[col_name][one_label] += 1
          else:
            # numerical column.

            # if empty, skip
            if not parsed_line[col_name].strip():
              continue

            numerical_results[col_name]['min'] = (
              min(numerical_results[col_name]['min'],
                  float(parsed_line[col_name])))
            numerical_results[col_name]['max'] = (
              max(numerical_results[col_name]['max'],
                  float(parsed_line[col_name])))
            numerical_results[col_name]['count'] += 1
            numerical_results[col_name]['sum'] += float(parsed_line[col_name])

  # Update numerical_results to just have min/min/mean
  for col_schema in schema_list:
    if col_schema['type'].lower() != 'string':
      col_name = col_schema['name']
      mean = numerical_results[col_name]['sum'] / numerical_results[col_name]['count']
      del numerical_results[col_name]['sum']
      del numerical_results[col_name]['count']
      numerical_results[col_name]['mean'] = mean

  # Write the numerical_results to a json file.
  file_io.write_string_to_file(
      os.path.join(args.output_dir, NUMERICAL_ANALYSIS_FILE),
      json.dumps(numerical_results, indent=2, separators=(',', ': ')))

  # Write the vocab files. Each label is on its own line.
  for name, label_count in six.iteritems(categorical_results):
    # Labels is now the string:
    # label1,count
    # label2,count
    # ...
    # where label1 is the most frequent label, and label2 is the 2nd most, etc.
    labels = '\n'.join(["%s,%d" % (label, count)
                        for label, count in sorted(six.iteritems(label_count),
                                                   key=lambda x: x[1],
                                                   reverse=True)])
    file_io.write_string_to_file(
        os.path.join(args.output_dir, CATEGORICAL_ANALYSIS_FILE % name),
        labels)



def get_transforms_per_input_column(schema, features):
  schema_name_to_transfrom = collections.defaultdict(set)
  for _, transform in six.iteritems(features):
    name = transform['source_column']
    schema_name_to_transfrom[name].add(transform['transform'])

  for col_schema in schema:
    col_name = col_schema['name']
    col_type = col_schema['type'].lower()
    if col_type in NUMERIC_SCHEMA:
      if not (schema_name_to_transfrom[col_name] <= set(NUMERIC_TRANSFORMS)):
        raise ValueError('Numerical schema columns can only have numerical transformations')
    elif col_type in STRING_SCHEMA:
      if not (
        (schema_name_to_transfrom[col_name] <= set(CATEGORICAL_TRANSFORMS))
        or (schema_name_to_transfrom[col_name] <= set(TEXT_TRANSFORMS))):
        raise ValueError('String schemas can exclusifly have categorical or text transformations')
    else:
      raise ValueError('Unknown schema type %s' col_type)

  return schema_name_to_transfrom



def expand_defaults(schema, features):
  """Add to features any default transformations.

  Not every column in the schema has an explicit feature transfromation listed
  in the featurs file. For these columns, add a default transformation based on
  the schema's type. The features dict is modified by this function call.
  """
  # Update source_column values
  for name, transform in six.iteritems(features):
    if not transform.get('source_column', None):
      transform['source_column'] = name

  columns_used = {x['source_column'] for x in six.itervalues(features)}
  schema_names = [x['name'] for x in schema]

  for source_column in columns_used:
    if source_column not in schema_names:
      raise ValueError('source column %s is not in the schema' % source_column)

  for name, transform in six.iteritems(features):
    if name in schema_names and name != transform['source_column']:
      raise ValueError(('%s is a schema name and it differs from the '
                        'source_column name %s') %
                       (name, transform['source_column']))

  # Update default transformation based on schema. 
  for col_schema in schema:
    schema_name = col_schema['name']
    schema_type = col_schema['type'].lower()

    if schema_type not in SUPPORTED_SCHEMA:
      raise ValueError('Only the following schema types are supported: %s' 
                        % ' '.join(SUPPORTED_SCHEMA))

    if schema_name not in columns_used:
      # add the default transform to the features
      if schema_type in NUMERIC_SCHEMA:
        features[schema_name] = {'transform': NUMERIC_TRANSFORMS[0],
                                 'source_column': schema_name}
      elif schema_type in STRING_SCHEMA:
        features[schema_name] = {'transform': CATEGORICAL_TRANSFORMS[0],
                                 'source_column': schema_name}
      else:
        raise NotImplementedError('Unknown type %s' % schema_type)      


def main(argv=None):
  args = parse_arguments(sys.argv if argv is None else argv)
  
  if args.csv_schema_file:
    schema = json.loads(file_io.read_file_to_string(args.csv_schema_file))
  else:
    import google.datalab.bigquery as bq
    schema = bq.Table(args.bigquery_table).schema._bq_schema
  features = json.loads(file_io.read_file_to_string(args.features_file))

  expand_defaults(schema, features) # features are updated.
  

  if args.cloud:
    run_cloud_analysis(args, schema, features)
  else:
    run_local_analysis(args, schema, features)

  #make_transform_graph(args, schema, features)


if __name__ == '__main__':
  main()
