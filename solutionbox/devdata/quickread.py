import tensorflow as tf

opt = tf.python_io.TFRecordOptions(
      compression_type=tf.python_io.TFRecordCompressionType.GZIP)
record_iterator = tf.python_io.tf_record_iterator(path='tfpreout/features_eval-00000-of-00001.tfrecord.gz', options=opt)

count = 0
for string_record in record_iterator:
    print('string_record')
    print('%r' % string_record)
    example = tf.train.Example()
    example.ParseFromString(string_record)
    print(example)
    count = count + 1
    if count > 5:
      break
