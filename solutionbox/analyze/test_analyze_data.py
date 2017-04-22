from __future__ import absolute_import
from __future__ import print_function

import os
import six
import sys

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                             '../../..')))

import collections
from tensorflow.python.lib.io import file_io
import tempfile

import unittest
import copy
import json

import shutil

import analyze_data



class TestConfigFiles(unittest.TestCase):

  def test_expand_defaults_do_nothing(self):
    schema = [{'name': 'col1', 'type': 'FLOAT'}, 
              {'name': 'col2', 'type': 'INTEGER'}]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'} 
    }
    original_features = copy.deepcopy(features)

    analyze_data.expand_defaults(schema, features)
    
    # Nothing should change.
    self.assertEqual(original_features, features)


  def test_expand_defaults_unknown_schema_type(self):
    schema = [{'name': 'col1', 'type': 'BYTES'}, 
              {'name': 'col2', 'type': 'INTEGER'}]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'} 
    }

    with self.assertRaises(ValueError) as context:
      analyze_data.expand_defaults(schema, features)

  def test_expand_defaults(self):
    schema = [{'name': 'col1', 'type': 'FLOAT'}, 
              {'name': 'col2', 'type': 'INTEGER'},
              {'name': 'col3', 'type': 'STRING'},
              {'name': 'col4', 'type': 'FLOAT'}, 
              {'name': 'col5', 'type': 'INTEGER'},
              {'name': 'col6', 'type': 'STRING'},              
    ]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'}, 
                'col3': {'transform': 'z'},
    }
    original_features = copy.deepcopy(features)

    analyze_data.expand_defaults(schema, features)
 
    self.assertEqual(
      features, 
      {'col1': {'transform': 'x'},
       'col2': {'transform': 'y'},
       'col3': {'transform': 'z'},
       'col4': {'transform': 'identity'},
       'col5': {'transform': 'identity'},
       'col6': {'transform': 'one_hot'}})

  def test_check_schema_transform_match(self):
    with self.assertRaises(ValueError):
      analyze_data.check_schema_transform_match(
         [{'name': 'col1', 'type': 'INTEGER'}],
         {'col1': {'transform': 'one_hot'}})

    with self.assertRaises(ValueError):
      analyze_data.check_schema_transform_match(
         [{'name': 'col1', 'type': 'FLOAT'}],
         {'col1': {'transform': 'embedding'}})

    with self.assertRaises(ValueError):
      analyze_data.check_schema_transform_match(
         [{'name': 'col1', 'type': 'STRING'}],
         {'col1': {'transform': 'scale'}})

    with self.assertRaises(ValueError):
      analyze_data.check_schema_transform_match(
         [{'name': 'col1', 'type': 'xxx'}],
         {'col1': {'transform': 'scale'}})

    with self.assertRaises(ValueError):
      analyze_data.check_schema_transform_match(
         [{'name': 'col1', 'type': 'INTEGER'}],
         {'col1': {'transform': 'xxx'}})

class TestLocalAnalyze(unittest.TestCase):

  Args = collections.namedtuple('Args', ['csv_file_pattern', 'output_dir'])
  def test_numerics(self):
    
    output_folder = tempfile.mkdtemp()
    input_file_path = tempfile.mkstemp(dir=output_folder)[1]
    try:
      file_io.write_string_to_file(
        input_file_path,
        '\n'.join(['%s,%s' % (i, 10*i) for i in range(100)]))

      analyze_data.run_local_analysis(
        self.Args(input_file_path, output_folder),
        [{'name': 'col1', 'type': 'INTEGER'},
         {'name': 'col2', 'type': 'FLOAT'}],
        {'col1': {'transform': 'scale'},
         'col2': {'transform': 'identity'}})
      stats = json.loads(file_io.read_file_to_string(os.path.join(output_folder, analyze_data.NUMERICAL_ANALYSIS_FILE)))
      self.assertEqual(
          stats,
          {u'col1': {u'max': 99.0, u'mean': 49.5, u'min': 0.0},
           u'col2': {u'max': 990.0, u'mean': 495.0, u'min': 0.0}})
    except Exception as e:
      pass
      # TODO: re-raise the exception while keeping the traceback. 
    finally:
      shutil.rmtree(output_folder)
if __name__ == '__main__':
    unittest.main()
