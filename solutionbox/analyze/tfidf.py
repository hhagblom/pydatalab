import tensorflow as tf
tf.InteractiveSession()

vocab_size = 3

def segment_indices(segment_ids):
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
  segment_lengths = tf.segment_sum(tf.ones_like(segment_ids), segment_ids)
  segment_starts = tf.gather(tf.concat([[0], tf.cumsum(segment_lengths)], 0),
                             segment_ids)
  return (tf.range(tf.size(segment_ids, out_type=segment_ids.dtype)) -
          segment_starts)

def _map_to_doc_contains_term(x):
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
  return next_tensor
  # Take the intermediar tensor and reduce over the term_index_in_doc
  # dimension. This produces a tensor with indices [<doc_id>, <term_id>]
  # and values [count_of_term_in_doc] and shape batch x vocab_size
  #term_count_per_doc = tf.sparse_reduce_sum_sparse(next_tensor, 1)


  #return term_count_per_doc
  #one_if_doc_contains_term = tf.SparseTensor(
  #    indices=term_count_per_doc.indices,
  #    values=tf.to_double(tf.greater(term_count_per_doc.values, 0)),
  #    dense_shape=term_count_per_doc.dense_shape)
  #return one_if_doc_contains_term

# red red red
# red green red
# blue
# blue blue

# red=0
# blue=1
# green=2
sp = tf.SparseTensor(indices=[[0, 0],
       [0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [3, 0],
       [3, 1]], values=[0, 0, 0, 0, 2, 0, 1, 1, 1], dense_shape=[4, 3])

next_tensor = _map_to_doc_contains_term(sp)
term_count_per_doc = tf.sparse_reduce_sum_sparse(next_tensor, 1)
one_if_doc_contains_term = tf.SparseTensor(
      indices=term_count_per_doc.indices,
      values=tf.to_double(tf.greater(term_count_per_doc.values, 0)),
      dense_shape=term_count_per_doc.dense_shape)

print('term_count_per_doc', term_count_per_doc.eval())


#                    0  1  2
reduced_term_freq = [2, 2, 1]
corpus_size=4


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

print('term_count_per_doc', term_count_per_doc.eval())
print('tfidf_weights', tfidf_weights.eval())
print('tfidf_ids', tfidf_ids.eval())

indices = tf.stack([term_count_per_doc.indices[:,0], segment_indices(term_count_per_doc.indices[:,0])], 1)
dense_shape = term_count_per_doc.dense_shape

new_tfidf_weights = tf.SparseTensor(indices=indices, values=tfidf_weights, dense_shape=dense_shape)
new_tfidf_ids = tf.SparseTensor(indices=indices, values=tfidf_ids, dense_shape=dense_shape)

current =  tf.SparseTensor(
    indices=sp.indices,
    values=idf_over_doc_size,
    dense_shape=sp.dense_shape)