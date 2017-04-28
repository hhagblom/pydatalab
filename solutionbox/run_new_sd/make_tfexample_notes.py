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


def main(argv=None):
 
  transform_fn_tmp_dir = 'pout/transform_fn'
  

  output_metadata = metadata_io.read_metadata('pout/transformed_metadata')
  input_metadata = metadata_io.read_metadata('pout/raw_metadata')
  
  g = tf.Graph()
  session = tf.Session(graph=g)
  with g.as_default():
    inputs, outputs = impl_helper.load_transform_fn_def('pout/transform_fn')

    # is this step needed?
    inputs = {key: inputs[key] for key in input_metadata.schema.column_schemas.keys()}
    outputs = {key: outputs[key] for key in output_metadata.schema.column_schemas.keys()}

    session.run(tf.tables_initializer())
    print(inputs)
    print(outputs)
    #feed_dict = impl_helper.make_feed_dict(inputs, tft_input_schema, self._batch)
    feed = {inputs['key']: [2,3],
            inputs['target']: ['misc.forsale', 'rec.sport.baseball'],
            inputs['text']: ['job least coupl start few him prove worth gee sound pretti reason kid play back role rather start everi day aaa talk point signific sampl rather hadn done anyth spring train caus manag whether minor leagu number real send him down until warm player readi big age player never',
                             'gaug free reign design own instrument cluster gaug choos beyond basic set turbo boost necessari turbo car fuel reserv warn level warn nice gaug cycl across differ sensor system such sensor altitud air flow love tranni diff brake temp great']
            }
    #feed = {inputs['num']: [2], inputs['cat']: ['dog']}
    result = session.run(outputs, feed_dict=feed)
    print('result')
    print(result)

if __name__ == '__main__':
  main()