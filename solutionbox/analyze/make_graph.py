
import tempfile

import tensorflow as tf
import tensorflow_transform as tft

from tensorflow.python.lib.io import file_io
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform import impl_helper
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.saved import constants




from tensorflow.contrib.session_bundle import bundle_shim
from tensorflow.contrib import lookup

def make_tft_input_schema():
  result = {
  		'num': tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'cat': tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''), 
  }

  return dataset_schema.from_feature_spec(result)

# A TITO function that scales x.
def _scale(x, min_x_value, max_x_value, output_min, output_max):
  return ((((x - min_x_value) * (output_max - output_min)) /
           (max_x_value - min_x_value)) + output_min)


# A TITO function that maps x into an int.
def _map_to_int(x, vocab):
  """Maps string tensor into indexes using vocab.
  Args:
    x : a Tensor/SparseTensor of string.
    vocab : a Tensor/SparseTensor containing unique string values within x.

  Returns:
    a Tensor/SparseTensor of indexes (int) of the same shape as x.
  """

  def _fix_vocab_if_needed(vocab):
    num_to_add = 1 - tf.minimum(tf.size(vocab), 1)
    return tf.concat([
        vocab, tf.fill(
            tf.reshape(num_to_add, (1,)), '__dummy_value__index_zero__')
    ], 0)

  table = lookup.string_to_index_table_from_tensor(
      _fix_vocab_if_needed(vocab), num_oov_buckets=0,
      default_value=-1)
  return table.lookup(x)



def make_preprocessing_fn():
  def preprocessing_fn(inputs):
    """User defined preprocessing function.

    Args:
      inputs: dictionary of input `tensorflow_transform.Column`.
    Returns:
      A dictionary of `tensorflow_transform.Column` representing the transformed
          columns.
    """
    result = {}

    min_x_value = 1.0
    max_x_value = 10
    output_min = -1.0
    output_max = 1.0
    result['scaled_num'] = tft.map(
        lambda x: _scale(x, min_x_value, max_x_value, output_min, output_max),
        inputs['num'])


    vocab = ['dog', 'fish', 'snake']
    result['cat'] = tft.map(
        lambda x: _map_to_int(x, vocab),
        inputs['cat'])
    return result

  return preprocessing_fn


def load_graph(model_path):
  session, meta_graph = bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(
      model_path, 
      tags=[constants.TRANSFORM_TAG])
  signature = meta_graph.signature_def[constants.TRANSFORM_SIGNATURE]
  inputs = {
      key: tensor_info_proto.name
      for (key, tensor_info_proto) in signature.inputs.items()
  }
  outputs = {
      key: tensor_info_proto.name
      for (key, tensor_info_proto) in signature.outputs.items()
  }
  return session, inputs, outputs


def main(argv=None):
  tft_input_schema = make_tft_input_schema()
  tft_input_metadata = dataset_metadata.DatasetMetadata(schema=tft_input_schema)
  preprocessing_fn = make_preprocessing_fn()

  # copy from /tft/beam/impl
  inputs, outputs = impl_helper.run_preprocessing_fn(preprocessing_fn, tft_input_schema)
  output_metadata = dataset_metadata.DatasetMetadata(schema=impl_helper.infer_feature_schema(outputs))


  transform_fn_tmp_dir = 'pout/transfrom_fn'
  input_columns_to_statistics = impl_helper.make_transform_fn_def(
      tft_input_schema, inputs, outputs, transform_fn_tmp_dir)


  metadata_io.write_metadata(output_metadata, 'pout/transformed_metadata')
  metadata_io.write_metadata(tft_input_metadata, 'pout/raw_metadata')
  print(transform_fn_tmp_dir)

  print('as reported in the in-memeory object.')
  print(inputs)
  print(outputs)


  print('from savedmodel')
  #session, sminputs, smoutputs = load_graph(transform_fn_tmp_dir)
  #print(sminputs)
  #print(smoutputs)
  g = tf.Graph()
  session = tf.Session(graph=g)
  with g.as_default():
    inputs, outputs = impl_helper.load_transform_fn_def('pout/transfrom_fn')

    # is this step needed?
    inputs = {key: inputs[key] for key in tft_input_metadata.schema.column_schemas.keys()}
    outputs = {key: outputs[key] for key in output_metadata.schema.column_schemas.keys()}

    session.run(tf.tables_initializer())
    print(inputs)
    print(outputs)
    #feed_dict = impl_helper.make_feed_dict(inputs, tft_input_schema, self._batch)
    feed = {inputs['num']: [2,3], inputs['cat']: ['dog', 'fish']}
    feed = {inputs['num']: [2], inputs['cat']: ['dog']}
    result = session.run(outputs, feed_dict=feed)
    print('result')
    print(result)



if __name__ == '__main__':
  main()