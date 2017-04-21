from __future__ import absolute_import
from __future__ import print_function

import os
import six
import sys

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                             '../../..')))


import unittest
import copy

import analyze_data



class TestConfigFiles(unittest.TestCase):

  def test_expand_defaults_source_column(self):
    schema = [{'name': 'col1', 'type': 'FLOAT'}, 
              {'name': 'col2', 'type': 'INTEGER'}]
    features = {'col1': {'transform': 'x'},
                'col2': {'transform': 'y'} 
    }
    original_features = copy.deepcopy(features)

    analyze_data.expand_defaults(schema, features)

    original_features['col1']['source_column'] = 'col1'
    original_features['col2']['source_column'] = 'col2'

    self.assertEqual(original_features, features)

  def test_expand_defaults_unknown_source_column(self):
    schema = [{'name': 'col1', 'type': 'FLOAT'}, 
              {'name': 'col2', 'type': 'INTEGER'}]
    features = {'col1': {'transform': 'x', 'source_column': 'col3'},
                'col2': {'transform': 'y'} 
    }

    with self.assertRaises(ValueError) as context:
      analyze_data.expand_defaults(schema, features)

    self.assertTrue('source column col3 is not in the schema' 
                    in context.exception)

  def test_expand_defaults_schema_name_reused(self):
    schema = [{'name': 'col1', 'type': 'FLOAT'}, 
              {'name': 'col2', 'type': 'INTEGER'}]
    # The problem here is that col1 is a schema name so there could be two
    # names for the same column now.
    features = {'col1': {'transform': 'x', 'source_column': 'col2'},
                'col2': {'transform': 'y'} 
    }

    with self.assertRaises(ValueError) as context:
      analyze_data.expand_defaults(schema, features)

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
      {'col1': {'source_column': 'col1', 'transform': 'x'},
       'col2': {'source_column': 'col2', 'transform': 'y'},
       'col3': {'source_column': 'col3', 'transform': 'z'},
       'col4': {'source_column': 'col4', 'transform': 'identity'},
       'col5': {'source_column': 'col5', 'transform': 'identity'},
       'col6': {'source_column': 'col6', 'transform': 'one_hot'}})

if __name__ == '__main__':
    unittest.main()
