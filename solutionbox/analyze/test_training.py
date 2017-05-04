from __future__ import absolute_import

import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from tensorflow.python.lib.io import file_io

def make_csv_data(filename, num_rows, problem_type, keep_target=True):
  """Writes csv data for preprocessing and training.

  Args:
    filename: writes data to local csv file.
    num_rows: how many rows of data will be generated.
    problem_type: 'classification' or 'regression'. Changes the target value.
    keep_target: if false, the csv file will have an empty column ',,' for the
        target.
  """
  random.seed(12321)

  def _drop_out(x):
    # Make 5% of the data missing
    if random.uniform(0, 1) < 0.05:
      return ''
    return x
  
  with open(filename, 'w') as f:
    for i in range(num_rows):
      num_id = random.randint(0, 20)
      num_scale = random.uniform(0, 30)

      str_one_hot = random.choice(['red', 'blue', 'green', 'pink', 'yellow', 'brown', 'black'])
      str_embedding = random.choice(['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr'])

      word_fn = lambda : random.choice(['car', 'truck', 'van', 'bike', 'train', 'drone'])

      str_bow =  [word_fn() for _ in range(random.randint(1,4))]
      str_tfidf =  [word_fn() for _ in range(random.randint(1,4))]

      color_map = {'red': 2, 'blue': 6, 'green': 4, 'pink': -5, 'yellow': -6, 'brown': -1, 'black': -7}
      abc_map = {'abc': -1, 'def': -1, 'ghi': 1, 'jkl': 1, 'mno': 2, 'pqr': 1}
      transport_map = {'car': 5, 'truck': 10, 'van': 15, 'bike': 20, 'train': -25, 'drone': -30}

      # Build some model.
      t = 0.5 + 0.5 * num_id - 2.5 * num_scale
      t += color_map[str_one_hot]
      t += abc_map[str_embedding]
      t += sum([transport_map[x] for x in str_bow])
      t += sum([transport_map[x]*0.5 for x in str_tfidf])

      if problem_type == 'classification':
        if t < 0:
          t = 100
        elif t < 20:
          t = 101
        else:
          t = 102

      str_bow = ' '.join(str_bow)
      str_tfidf = ' '.join(str_tfidf)

      num_id = _drop_out(num_id)
      num_scale = _drop_out(num_scale)
      str_one_hot = _drop_out(str_one_hot)
      str_embedding = _drop_out(str_embedding)
      str_bow = _drop_out(str_bow)
      str_tfidf = _drop_out(str_tfidf)

      if keep_target:
          csv_line = "{key},{target},{num_id},{num_scale},{str_one_hot},{str_embedding},{str_bow},{str_tfidf}\n".format(
            key=i,
            target=t,
            num_id=num_id,
            num_scale=num_scale,
            str_one_hot=str_one_hot,
            str_embedding=str_embedding,
            str_bow=str_bow,
            str_tfidf=str_tfidf)
      else:
          csv_line = "{key},{num_id},{num_scale},{str_one_hot},{str_embedding},{str_bow},{str_tfidf}\n".format(
            key=i,
            num_id=num_id,
            num_scale=num_scale,
            str_one_hot=str_one_hot,
            str_embedding=str_embedding,
            str_bow=str_bow,
            str_tfidf=str_tfidf)
      f.write(csv_line)


class TestTrainer(unittest.TestCase):
  """Tests 
  """
  def __init__(self, *args, **kwargs):
    super(TestTrainer, self).__init__(*args, **kwargs)

    # Allow this class to be subclassed for quick tests that only care about
    # training working, not model loss/accuracy.
    self._max_steps = 2500
    self._check_model_fit = True

    # Log everything
    self._logger = logging.getLogger('TestStructuredDataLogger')
    self._logger.setLevel(logging.DEBUG)
    if not self._logger.handlers:
      self._logger.addHandler(logging.StreamHandler(stream=sys.stdout))

  def setUp(self):
    #self._test_dir = tempfile.mkdtemp()
    self._test_dir = './tmp'
    self._analysis_output = os.path.join(self._test_dir, 'analysis_output')
    self._transform_output = os.path.join(self._test_dir, 'transform_output')
    self._train_output = os.path.join(self._test_dir, 'train_output')

    file_io.recursive_create_dir(self._analysis_output)
    file_io.recursive_create_dir(self._transform_output)
    file_io.recursive_create_dir(self._train_output)

    self._csv_train_filename = os.path.join(self._test_dir, 'train_csv_data.csv')
    self._csv_eval_filename = os.path.join(self._test_dir, 'eval_csv_data.csv')
    self._csv_predict_filename = os.path.join(self._test_dir, 'predict_csv_data.csv')
    self._schema_filename = os.path.join(self._test_dir, 'schema_file.json')
    self._features_filename = os.path.join(self._test_dir, 'features_file.json')

  def tearDown(self):
    pass
    #self._logger.debug('TestTrainer: removing test dir ' + self._test_dir)
    #shutil.rmtree(self._test_dir)

  def _run_analyze(self, problem_type):
    features = {
        'num_id': {'transform': 'identity'},
        'num_scale': {'transform': 'scale', 'value': 4},
        'str_one_hot': {'transform': 'one_hot'},
        'str_embedding': {'transform': 'embedding', 'embedding_dim': 3},
        'str_bow': {'transform': 'bag_of_words'}, 
        'str_tfidf': {'transform': 'tfidf'}, 
        'target': {'transform': 'target'},
        'key': {'transform': 'key'},
    }

    schema = [
        {'name': 'key', 'type': 'integer'},
        {'name': 'target', 'type': 'string'},
        {'name': 'num_id', 'type': 'integer'},
        {'name': 'num_scale', 'type': 'float'},
        {'name': 'str_one_hot', 'type': 'string'},
        {'name': 'str_embedding', 'type': 'string'},
        {'name': 'str_bow', 'type': 'string'},
        {'name': 'str_tfidf', 'type': 'string'},
    ]

    file_io.write_string_to_file(self._schema_filename, json.dumps(schema, indent=2))
    file_io.write_string_to_file(self._features_filename, json.dumps(features, indent=2))

    make_csv_data(self._csv_train_filename, 200, problem_type, True)
    make_csv_data(self._csv_eval_filename, 100, problem_type, True)
    make_csv_data(self._csv_predict_filename, 100, problem_type, False)

    cmd = ['python analyze_data.py',
           '--output-dir=' + self._analysis_output,
           '--csv-file-pattern=' + self._csv_train_filename,
           '--csv-schema-file=' + self._schema_filename,
           '--features-file=' + self._features_filename,
    ]
  
    subprocess.check_call(' '.join(cmd), shell=True)

  def _run_transform(self):
    cmd = ['python transform_to_tfexample.py',
           '--csv-file-pattern=' + self._csv_train_filename,
           '--analyze-output-dir=' + self._analysis_output,
           '--output-filename-prefix=featrues_train',
           '--output-dir=' + self._transform_output,
           '--shuffle',
           '--target']
    subprocess.check_call(' '.join(cmd), shell=True)
   
    cmd = ['python transform_to_tfexample.py',
           '--csv-file-pattern=' + self._csv_eval_filename,
           '--analyze-output-dir=' + self._analysis_output,
           '--output-filename-prefix=featrues_eval',
           '--output-dir=' + self._transform_output,
           '--target']
    subprocess.check_call(' '.join(cmd), shell=True)   

    cmd = ['python transform_to_tfexample.py',
           '--csv-file-pattern=' + self._csv_predict_filename,
           '--analyze-output-dir=' + self._analysis_output,
           '--output-filename-prefix=featrues_predict',
           '--output-dir=' + self._transform_output,
           '--no-target']
    subprocess.check_call(' '.join(cmd), shell=True) 

  def _run_training(self, problem_type, model_type, extra_args=[]):
    """Runs training.

    Args:
      problem_type: 'regression' or 'classification'
      model_type: 'linear' or 'dnn'
      transform: JSON object of the transforms file.
      extra_args: list of strings to pass to the trainer.
    """
    pass


  
  def testClassificationLinear(self):
    self._logger.debug('\n\nTesting classification Linear')

    problem_type='classification',
    model_type='linear'
    self._run_analyze(problem_type)
    self._run_transform()

    self._run_training(problem_type='classification',
                       model_type='linear')



if __name__ == '__main__':
    unittest.main()