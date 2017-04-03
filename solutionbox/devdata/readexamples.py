import argparse
import os
import re
import sys

# run via
# python -m trainer/task --train-data-paths train_csv_data.csv  --eval-data-paths eval_csv_data.csv  --job-dir TOUT --preprocess-output-dir x --transforms-file y --model-type dnn_classification --top-n 3


#from . import util
import tensorflow as tf

#from tensorflow_transform.saved import input_fn_maker
#from tensorflow_transform.tf_metadata import metadata_io

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.lib.io import file_io


def gzip_reader_fn():
  return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def get_reader_input_fn(data_paths, batch_size, shuffle, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      [data_paths], num_epochs=num_epochs, shuffle=shuffle)
  print('filename_queue')
  print(filename_queue)



  #reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
  reader = gzip_reader_fn()
  #_, rows = reader.read(filename_queue)
  _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

  # Parse the CSV File
  features =  {
  	'key': tf.FixedLenFeature([], dtype=tf.string), 
    'target': tf.FixedLenFeature([], dtype=tf.int64),
    'num1': tf.FixedLenFeature([], dtype=tf.float32),
    'num2': tf.FixedLenFeature([], dtype=tf.float32),
    'num3': tf.FixedLenFeature([], dtype=tf.float32),
    'str1': tf.FixedLenFeature([], dtype=tf.int64),
    'str2': tf.FixedLenFeature([], dtype=tf.int64),
    'str3': tf.FixedLenFeature([], dtype=tf.int64),
  }
  features = tf.parse_example(rows, features=features)


  return rows, features, features.pop('target')	

def xxxxget_reader_input_fn(data_paths, batch_size, shuffle, num_epochs=None):
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





coord = tf.train.Coordinator(clean_stop_exception_types=(
        tf.errors.CancelledError, tf.errors.OutOfRangeError))

# make ()
rows, features, labels = get_reader_input_fn('tfpreout/features_eval-00000-of-00001.tfrecord.gz', 10, False, None)

with tf.Session() as sess:
  sess.run( [tf.tables_initializer(),
        tf.local_variables_initializer(), tf.global_variables_initializer()])
  tf.train.start_queue_runners(coord=coord, sess=sess)

  ans = sess.run(rows)
  print(ans)

  ##ans = sess.run(features)
  #print(ans)
