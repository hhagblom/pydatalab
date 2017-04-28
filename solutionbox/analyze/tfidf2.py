import argparse
import collections
import csv
import json
import os
import six
import sys

import pandas as pd


import tensorflow_transform as tft
import tensorflow as tf

from tensorflow.python.lib.io import file_io
from tensorflow.contrib import lookup

from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform import impl_helper

sess = tf.InteractiveSession()
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
  segment_lengths = tf.unsorted_segment_sum(tf.ones_like(segment_ids), segment_ids, tf.to_int32(num_segments))
  segment_starts = tf.gather(tf.concat([[0], tf.cumsum(segment_lengths)], 0),
                             segment_ids)
  return (tf.range(tf.size(segment_ids, out_type=segment_ids.dtype)) -
          segment_starts)

def get_term_count_per_doc(x, vocab_size):
  """Creates a SparseTensor with 1s at every doc/term pair index.

  Args:
    x : a SparseTensor of int64 representing string indices in vocab.

  Returns:
    a SparseTensor with 1s at indices <doc_index_in_batch>,
        <term_index_in_vocab> for every term/doc pair.
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
  """

  Args:
    vocab: list of strings. Must include '' in the list.
    example_count: example_count[i] is the number of examples that contain the
      token vocab[i]
    corpus_size: how many examples there are.
  """

  def _get_tfidf_weights():
    # Add one to the reduced term freqnencies to avoid dividing by zero.
    idf = tf.log(tf.to_double(corpus_size) / (
        1.0 + tf.to_double(reduced_term_freq)))

    dense_doc_sizes = tf.to_double(tf.sparse_reduce_sum(tf.SparseTensor(
        indices=sp.indices,
        values=tf.ones_like(sp.values),
        dense_shape=sp.dense_shape), 1))

    # For every term in x, divide the idf by the doc size.
    # The two gathers both result in shape <sum_doc_sizes>
    idf_over_doc_size = (tf.gather(idf, sp.values) /
                         tf.gather(dense_doc_sizes, sp.indices[:, 0]))

    tfidf_weights = (tf.multiply(
                            tf.gather(idf, term_count_per_doc.indices[:,1]),
                            tf.to_double(term_count_per_doc.values))  /
                         tf.gather(dense_doc_sizes, term_count_per_doc.indices[:, 0]))
    tfidf_ids = term_count_per_doc.indices[:,1]

  def _tfidf(x):
    split = tf.string_split(x)
    table = lookup.string_to_index_table_from_tensor(
        vocab, num_oov_buckets=0,
        default_value=vocab.index(''))
    int_text = table.lookup(split)

    #SparseTensorValue(indices=array([[0, 0],
    #   [1, 0],
    #   [1, 2],
    #   [2, 1],
    #   [3, 1]]), values=array([3, 2, 1, 1, 2], dtype=int32), dense_shape=array([4, 3]))
    term_count_per_doc = get_term_count_per_doc(int_text, len(vocab))

    # Add one to the reduced term freqnencies to avoid dividing by zero.
    idf = tf.log(tf.to_double(corpus_size) / ( 1.0 + tf.to_double(example_count)))

    dense_doc_sizes = tf.to_double(tf.sparse_reduce_sum(tf.SparseTensor(
        indices=int_text.indices,
        values=tf.ones_like(int_text.values),
        dense_shape=int_text.dense_shape), 1))

    tfidf_weights = (tf.multiply(
                        tf.gather(idf, term_count_per_doc.indices[:,1]),
                        tf.to_double(term_count_per_doc.values))  /
                     tf.gather(dense_doc_sizes, term_count_per_doc.indices[:, 0]))
    sess.run(tf.tables_initializer())
    print('dense_doc_sizes', dense_doc_sizes.eval())
    print('term_count_per_doc', term_count_per_doc.eval())
    print('tdf', idf.eval())
    tfidf_ids = term_count_per_doc.indices[:,1]

    indices = tf.stack([term_count_per_doc.indices[:,0], 
                        segment_indices(term_count_per_doc.indices[:,0], int_text.dense_shape[0])],
                       1)
    dense_shape = term_count_per_doc.dense_shape

    tfidf_st_weights = tf.SparseTensor(indices=indices, values=tfidf_weights, dense_shape=dense_shape)
    tfidf_st_ids = tf.SparseTensor(indices=indices, values=tfidf_ids, dense_shape=dense_shape)            

    #if part == 'ids':
    #  return tfidf_st_ids
    #else:
    #  return tfidf_st_weights
    return [tfidf_st_weights, tfidf_st_ids]

  return _tfidf
# red red red
# red green red
# blue
# blue blue

# red=0
# blue=1
# green=2


# red red red
# red green red
# blue
# blue blue

# pizza=0
# ice_cream=1
# cookies=2
     #       0        1       2   3    4       5
#vocab = ['red', 'blue', 'green', '']
#x = ['red red red', 'red green red', 'blue', 'blue blue']
#example_count = [2, 2, 1, 0]
#corpus_size=len(x)
#part='ids'


vocab = ['red', 'blue', 'green', '']
x = ['red', '', 'color', 'blue blue']
example_count = [2, 2, 1, 0]
corpus_size=len(x)
part='ids'


tfmap = make_tfidf_tito(vocab, example_count, corpus_size, part)
ans = tfmap(x)
#sess.run(tf.tables_initializer())

print('ids', ans[1].eval())
print('weights', ans[0].eval())

