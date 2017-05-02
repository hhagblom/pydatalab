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
import csv
import json
import os
import sys
import pandas as pd
import six
import tensorflow as tf
import tensorflow_transform as tft

from tensorflow.contrib import lookup
from tensorflow.python.lib.io import file_io
from tensorflow_transform import impl_helper
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io

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

INTEGER_SCHEMA = 'integer'
FLOAT_SCHEMA = 'float'
STRING_SCHEMA = 'string'
NUMERIC_SCHEMA = [INTEGER_SCHEMA, FLOAT_SCHEMA]


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
                      help=('Input CSV file names. May contain a file pattern. '
                            'File prefix must include absolute file path.'))
  parser.add_argument('--csv-schema-file',
                      type=str,
                      required=False,
                      help=('BigQuery json schema file'))

  # If using bigquery table
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

  if not ((args.bigquery_table and args.csv_file_pattern is None and
           args.csv_schema_file is None) or
          (args.bigquery_table is None and args.csv_file_pattern and
           args.csv_schema_file)):
    raise ValueError('either --csv-schema-file and --csv-file-pattern must both'
                     ' be set or just --bigquery-table is set')

  return args


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# start of Tensor In Tensor Out (TITO) fuctions
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def make_identity_tito(x):
  def _id(x):
    return tf.identity(x)
  return _id


def make_scale_tito(min_x_value, max_x_value, output_min, output_max):
  def _scale(x):
    min_x_valuef = tf.to_float(min_x_value)
    max_x_valuef = tf.to_float(max_x_value)
    output_minf = tf.to_float(output_min)
    output_maxf = tf.to_float(output_max)
    return ((((tf.to_float(x) - min_x_valuef) * (output_maxf - output_minf)) /
            (max_x_valuef - min_x_valuef)) + output_minf)

  return _scale


def make_str_to_int_tito(vocab, default_value):
  def _str_to_int(x):
    table = lookup.string_to_index_table_from_tensor(
        vocab, num_oov_buckets=0,
        default_value=default_value)
    return table.lookup(x)
  return _str_to_int


def segment_indices(segment_ids, num_segments):
  """Returns a tensor of indices within each segment.

  segment_ids should be a sequence of non-decreasing non-negative integers that
  define a set of segments, e.g. [0, 0, 1, 2, 2, 2] defines 3 segments of length
  2, 1 and 3.  The return value is a tensor containing the indices within each
  segment.

  Example input: [0, 0, 1, 2, 2, 2]
  Example output: [0, 1, 0, 0, 1, 2]

  Args:
    segment_ids: A 1-d tensor containing an non-decreasing sequence of
        non-negative integers with type `tf.int32` or `tf.int64`.

  Returns:
    A tensor containing the indices within each segment.
  """
  segment_lengths = tf.unsorted_segment_sum(tf.ones_like(segment_ids),
                                            segment_ids,
                                            tf.to_int32(num_segments))
  segment_starts = tf.gather(tf.concat([[0], tf.cumsum(segment_lengths)], 0),
                             segment_ids)
  return (tf.range(tf.size(segment_ids, out_type=segment_ids.dtype)) -
          segment_starts)


def get_term_count_per_doc(x, vocab_size):
  """Creates a SparseTensor with 1s at every doc/term pair index.

  Args:
    x : a SparseTensor of int64 representing string indices in vocab.

  Returns:
    a SparseTensor with count at indices <doc_index_in_batch>,
        <term_index_in_vocab> for every term/doc pair. Example: the tensor
        SparseTensorValue(
          indices=array([[0, 0],
                         [1, 0],
                         [1, 2],
                         [2, 1],
                         [3, 1]]),
          values=array([3, 8, 9, 3, 4], dtype=int32),
          dense_shape=array([4, 3]))
        says the 2nd example/document (row index 1) has two tokens, and
        token 0 occures 8 times and token 2 occures 9 times.
  """
  # Construct intermediary sparse tensor with indices
  # [<doc>, <term_index_in_doc>, <vocab_id>] and tf.ones values.
  split_indices = tf.to_int64(
      tf.split(x.indices, axis=1, num_or_size_splits=2))
  expanded_values = tf.to_int64(tf.expand_dims(x.values, 1))
  next_index = tf.concat(
      [split_indices[0], split_indices[1], expanded_values], axis=1)
  next_values = tf.ones_like(x.values)
  vocab_size_as_tensor = tf.constant([vocab_size], dtype=tf.int64)
  next_shape = tf.concat(
      [x.dense_shape, vocab_size_as_tensor], 0)
  next_tensor = tf.SparseTensor(
      indices=tf.to_int64(next_index),
      values=next_values,
      dense_shape=next_shape)

  # Take the intermediar tensor and reduce over the term_index_in_doc
  # dimension. This produces a tensor with indices [<doc_id>, <term_id>]
  # and values [count_of_term_in_doc] and shape batch x vocab_size
  term_count_per_doc = tf.sparse_reduce_sum_sparse(next_tensor, 1)
  return term_count_per_doc


def make_tfidf_tito(vocab, example_count, corpus_size, part):
  """Make Term Frequency - Inverse Document Frequency transfrom.

  TF(term, doc) := count of 'term' in doc / numer of terms in doc
  IDF(term) := log(corpus_size/(1 + number of documents that contain 'term'))

  Args:
    vocab: list of strings. Must include '' in the list.
    example_count: example_count[i] is the number of examples that contain the
      token vocab[i]
    corpus_size: how many examples there are.
    part: 'ids' or 'weights'. Returns the weights or ids of the transform.

  Returns:

  """
  def _tfidf(x):
    split = tf.string_split(x)
    table = lookup.string_to_index_table_from_tensor(
        vocab, num_oov_buckets=0,
        default_value=len(vocab))
    int_text = table.lookup(split)

    term_count_per_doc = get_term_count_per_doc(int_text, len(vocab) + 1)

    # Add one to the reduced term freqnencies to avoid dividing by zero.
    example_count_with_oov = tf.to_double(tf.concat([example_count, [0]], 0))
    idf = tf.log(tf.to_double(corpus_size) / (1.0 + example_count_with_oov))

    dense_doc_sizes = tf.to_double(tf.sparse_reduce_sum(tf.SparseTensor(
        indices=int_text.indices,
        values=tf.ones_like(int_text.values),
        dense_shape=int_text.dense_shape), 1))

    idf_times_term_count = tf.multiply(
        tf.gather(idf, term_count_per_doc.indices[:, 1]),
        tf.to_double(term_count_per_doc.values))
    tfidf_weights = (
        idf_times_term_count / tf.gather(dense_doc_sizes,
                                         term_count_per_doc.indices[:, 0]))

    tfidf_ids = term_count_per_doc.indices[:, 1]

    indices = tf.stack([term_count_per_doc.indices[:, 0],
                        segment_indices(term_count_per_doc.indices[:, 0],
                                        int_text.dense_shape[0])],
                       1)
    dense_shape = term_count_per_doc.dense_shape

    tfidf_st_weights = tf.SparseTensor(indices=indices,
                                       values=tfidf_weights,
                                       dense_shape=dense_shape)
    tfidf_st_ids = tf.SparseTensor(indices=indices,
                                   values=tfidf_ids,
                                   dense_shape=dense_shape)

    if part == 'ids':
      return tfidf_st_ids
    else:
      return tfidf_st_weights

  return _tfidf


def make_bag_of_words_tito(vocab, part):
  def _bow(x):
    split = tf.string_split(x)
    table = lookup.string_to_index_table_from_tensor(
        vocab, num_oov_buckets=0,
        default_value=len(vocab))
    int_text = table.lookup(split)

    term_count_per_doc = get_term_count_per_doc(int_text, len(vocab) + 1)

    bow_weights = tf.to_float(term_count_per_doc.values)
    bow_ids = term_count_per_doc.indices[:, 1]

    indices = tf.stack([term_count_per_doc.indices[:, 0],
                        segment_indices(term_count_per_doc.indices[:, 0],
                                        int_text.dense_shape[0])],
                       1)
    dense_shape = term_count_per_doc.dense_shape

    bow_st_weights = tf.SparseTensor(indices=indices, values=bow_weights, dense_shape=dense_shape)
    bow_st_ids = tf.SparseTensor(indices=indices, values=bow_ids, dense_shape=dense_shape)

    if part == 'ids':
      return bow_st_ids
    else:
      return bow_st_weights

  return _bow


def make_preprocessing_fn(args, features):
  def preprocessing_fn(inputs):
    """TFT preprocessing function.

    Args:
      inputs: dictionary of input `tensorflow_transform.Column`.
    Returns:
      A dictionary of `tensorflow_transform.Column` representing the transformed
          columns.
    """
    stats = {}
    if os.path.isfile(os.path.join(args.output_dir, STATS_FILE)):
      stats = json.loads(file_io.read_file_to_string(os.path.join(args.output_dir, STATS_FILE)))

    result = {}
    for name, transform in six.iteritems(features):
      transform_name = transform['transform']

      if transform_name == KEY_TRANSFORM:
        transform_name = 'identity'
      elif transform_name == TARGET_TRANSFORM:
        if os.path.isfile(os.path.join(args.output_dir, VOCAB_ANALYSIS_FILE % name)):
          transform_name = 'one_hot'
        else:
          transform_name = 'identity'

      if transform_name == 'identity':
        result[name] = inputs[name]
      elif transform_name == 'scale':
        result[name] = tft.map(
            make_scale_tito(min_x_value=stats['column_stats'][name]['min'],
                            max_x_value=stats['column_stats'][name]['max'],
                            output_min=transform.get('value', 1) * (-1),
                            output_max=transform.get('value', 1)),
            inputs[name])
      elif transform_name in [ONE_HOT_TRANSFORM, EMBEDDING_TRANSFROM,
                              TFIDF_TRANSFORM, BOW_TRANSFORM]:
        vocab_str = file_io.read_file_to_string(
            os.path.join(args.output_dir, VOCAB_ANALYSIS_FILE % name))
        vocab_pd = pd.read_csv(six.StringIO(vocab_str),
                               header=None,
                               names=['vocab', 'count'])
        vocab = vocab_pd['vocab'].tolist()
        ex_count = vocab_pd['count'].tolist()

        if transform_name == TFIDF_TRANSFORM:
          result[name + '_ids'] = tft.map(
              make_tfidf_tito(vocab=vocab,
                              example_count=ex_count,
                              corpus_size=stats['num_examples'],
                              part='ids'),
              inputs[name])
          result[name + '_weights'] = tft.map(
              make_tfidf_tito(vocab=vocab,
                              example_count=ex_count,
                              corpus_size=stats['num_examples'],
                              part='weights'),
              inputs[name])
        elif transform_name == BOW_TRANSFORM:
          result[name + '_ids'] = tft.map(
              make_bag_of_words_tito(vocab=vocab, part='ids'),
              inputs[name])
          result[name + '_weights'] = tft.map(
              make_bag_of_words_tito(vocab=vocab, part='weights'),
              inputs[name])
        else:
          result[name] = tft.map(make_str_to_int_tito(vocab, len(vocab)),
                                 inputs[name])
      elif transform_name == KEY_TRANSFORM:
          result[name] = tft.map(make_identity_tito(), inputs[name])
      else:
        raise ValueError('unknown transform %s' % transform_name)
    return result

  return preprocessing_fn


def make_tft_input_schema(schema, args):
  """Make a tft-stype schema file.

  In the tft tramework, this is where default values are recoreded for training.

  Args:
    schema: schema file
    args: command line args
  """
  result = {}

  # stats file us used to get default values.
  stats = {}
  if file_io.file_exists(os.path.join(args.output_dir, STATS_FILE)):
    stats = json.loads(
        file_io.read_file_to_string(os.path.join(args.output_dir, STATS_FILE)))

  for col_schema in schema:
    col_type = col_schema['type'].lower()
    col_name = col_schema['name']
    if col_type == INTEGER_SCHEMA:
      default_value = stats.get('column_stats').get(col_name, {}).get('mean', 0)
      result[col_name] = tf.FixedLenFeature(
          shape=[],
          dtype=tf.int64,
          default_value=int(default_value))
    elif col_type == FLOAT_SCHEMA:
      default_value = stats.get('column_stats').get(col_name, {}).get('mean', 0)
      result[col_name] = tf.FixedLenFeature(
          shape=[],
          dtype=tf.float32,
          default_value=float(default_value))
    elif col_type == STRING_SCHEMA:
      result[col_name] = tf.FixedLenFeature(shape=[],
                                            dtype=tf.string,
                                            default_value='')
    else:
      raise ValueError('Unknown schema type %s' % col_type)

  return dataset_schema.from_feature_spec(result)


def make_transform_graph(args, schema, features):
  """Writes a tft transform fn, and metadata files.

  Args:
    args: command line args
    schema: schema file
    features: features file
  """

  tft_input_schema = make_tft_input_schema(schema, args)
  tft_input_metadata = dataset_metadata.DatasetMetadata(schema=tft_input_schema)
  preprocessing_fn = make_preprocessing_fn(args, features)

  # copy from /tft/beam/impl
  inputs, outputs = impl_helper.run_preprocessing_fn(
      preprocessing_fn=preprocessing_fn,
      schema=tft_input_schema)
  output_metadata = dataset_metadata.DatasetMetadata(
      schema=impl_helper.infer_feature_schema(outputs))

  transform_fn_dir = os.path.join(args.output_dir, TRANSFORM_FN_DIR)

  # This writes the SavedModel
  impl_helper.make_transform_fn_def(
      schema=tft_input_schema,
      inputs=inputs,
      outputs=outputs,
      saved_model_dir=transform_fn_dir)

  metadata_io.write_metadata(
      metadata=output_metadata,
      path=os.path.join(args.output_dir, TRANSFORMED_METADATA_DIR))
  metadata_io.write_metadata(
      metadata=tft_input_metadata,
      path=os.path.join(args.output_dir, RAW_METADATA_DIR))


def run_cloud_analysis(args, schema, features):
  pass


def run_local_analysis(args, schema, features):
  """Use pandas analyze csv files.

  Produces a stats file and vocab files.

  Args:
    args: commmand line args
    schema: BQ schema file
    features: featurs file.

  Raises:
    ValueError: on unknown transfrorms/schemas
  """
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

  num_examples = 0
  # for each file, update the numerical stats from that file, and update the set
  # of unique labels.
  for input_file in input_files:
    with file_io.FileIO(input_file, 'r') as f:
      for line in csv.reader(f):
        parsed_line = dict(zip(header, line))
        num_examples += 1

        for col_schema in schema:
          col_name = col_schema['name']
          col_type = col_schema['type'].lower()
          transform = features[col_name]['transform']

          # Map the target transfrom into one_hot or identity.
          if transform == TARGET_TRANSFORM:
            if col_type == STRING_SCHEMA:
              transform = ONE_HOT_TRANSFORM
            elif col_type in NUMERIC_SCHEMA:
              transform = IDENTITY_TRANSFORM
            else:
              raise ValueError('Unknown schema type')

          if transform in TEXT_TRANSFORMS:
            split_strings = parsed_line[col_name].split(' ')

            # If a label is in the row N times, increase it's vocab count by 1.
            for one_label in set(split_strings):
              # Filter out empty strings
              if one_label:
                vocabs[col_name][one_label] += 1
          elif transform in CATEGORICAL_TRANSFORMS:
            if parsed_line[col_name]:
              vocabs[col_name][parsed_line[col_name]] += 1
          elif transform in NUMERIC_TRANSFORMS:
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
          elif transform == KEY_TRANSFORM:
            pass
          else:
            raise ValueError('Unknown transform %s' % transform)

  # Write the vocab files. Each label is on its own line.
  vocab_sizes = {}
  for name, label_count in six.iteritems(vocabs):
    # Labels is now the string:
    # label1,count
    # label2,count
    # ...
    # where label1 is the most frequent label, and label2 is the 2nd most, etc.
    labels = '\n'.join(['%s,%d' % (label, count)
                        for label, count in sorted(six.iteritems(label_count),
                                                   key=lambda x: x[1],
                                                   reverse=True)])
    file_io.write_string_to_file(
        os.path.join(args.output_dir, VOCAB_ANALYSIS_FILE % name),
        labels)

    vocab_sizes[name] = {'vocab_size': len(label_count)}

  # Update numerical_results to just have min/min/mean
  for col_name in numerical_results:
    mean = (numerical_results[col_name]['sum'] /
            float(numerical_results[col_name]['count']))
    del numerical_results[col_name]['sum']
    del numerical_results[col_name]['count']
    numerical_results[col_name]['mean'] = mean

  # Write the stats file.
  numerical_results.update(vocab_sizes)
  stats = {'column_stats': numerical_results, 'num_examples': num_examples}
  file_io.write_string_to_file(
      os.path.join(args.output_dir, STATS_FILE),
      json.dumps(stats, indent=2, separators=(',', ': ')))


def check_schema_transform_match(schema, features):
  """Checks that the transform and schema do not conflict.

  Args:
    schema: schema file
    features: features file

  Raises:
    ValueError if transform cannot be applied given schema type.
  """
  num_key_transforms = 0
  num_target_transforms = 0

  for col_schema in schema:
    col_name = col_schema['name']
    col_type = col_schema['type'].lower()

    transform = features[col_name]['transform']
    if transform == KEY_TRANSFORM:
      num_key_transforms += 1
      continue
    elif transform == TARGET_TRANSFORM:
      num_target_transforms += 1
      continue

    if col_type in NUMERIC_SCHEMA:
      if transform not in NUMERIC_TRANSFORMS:
        raise ValueError(
            'Transform %s not supported by schema %s' % (transform, col_type))
    elif col_type == STRING_SCHEMA:
      if transform not in CATEGORICAL_TRANSFORMS + TEXT_TRANSFORMS:
        raise ValueError(
            'Transform %s not supported by schema %s' % (transform, col_type))
    else:
      raise ValueError('Unsupported schema type %s' % col_type)

  if num_key_transforms != 1 or num_target_transforms != 1:
    raise ValueError('Must have exactly one key and target transform')


def expand_defaults(schema, features):
  """Add to features any default transformations.

  Not every column in the schema has an explicit feature transformation listed
  in the featurs file. For these columns, add a default transformation based on
  the schema's type. The features dict is modified by this function call.

  Args:
    schema: schema file
    features: features file

  Raises:
    ValueError: if transform cannot be applied given schema type.
  """

  schema_names = [x['name'] for x in schema]

  for source_column in six.iterkeys(features):
    if source_column not in schema_names:
      raise ValueError('source column %s is not in the schema' % source_column)

  # Update default transformation based on schema.
  for col_schema in schema:
    schema_name = col_schema['name']
    schema_type = col_schema['type'].lower()

    if schema_type not in NUMERIC_SCHEMA + [STRING_SCHEMA]:
      raise ValueError(('Only the following schema types are supported: %s'
                        % ' '.join(NUMERIC_SCHEMA + [STRING_SCHEMA])))

    if schema_name not in six.iterkeys(features):
      # add the default transform to the features
      if schema_type in NUMERIC_SCHEMA:
        features[schema_name] = {'transform': NUMERIC_TRANSFORMS[0]}
      elif schema_type == STRING_SCHEMA:
        features[schema_name] = {'transform': CATEGORICAL_TRANSFORMS[0]}
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

  expand_defaults(schema, features)  # features are updated.
  check_schema_transform_match(schema, features)

  if args.cloud:
    run_cloud_analysis(args, schema, features)
  else:
    if not os.path.isdir(args.output_dir):
      os.makedirs(args.output_dir)
    run_local_analysis(args, schema, features)

  # Also writes the transform fn and tft metadata.
  make_transform_graph(args, schema, features)

  # Save a copy of the schema and features in the output folder.
  file_io.write_string_to_file(
    os.path.join(args.output_dir, SCHEMA_FILE),
    json.dumps(schema, indent=2))

  file_io.write_string_to_file(
    os.path.join(args.output_dir, FEATURES_FILE),
    json.dumps(features, indent=2))


if __name__ == '__main__':
  main()