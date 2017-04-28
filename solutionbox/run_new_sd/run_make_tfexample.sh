#! /bin/bash

# 18 seconds
rm -fr exout
time python make_tfexample.py  \
	--csv-file-pattern news_clean_train.csv \
	--analyze-output-dir pout/ \
	--output-dir exout \
	--output-filename-prefix features_train

time python make_tfexample.py  \
	--csv-file-pattern news_clean_test.csv \
	--analyze-output-dir pout/ \
	--output-dir exout \
	--output-filename-prefix features_test