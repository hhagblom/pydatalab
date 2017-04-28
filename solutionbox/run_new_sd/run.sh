#! /bin/bash
python ../analyze/analyze_data.py \
 --output-dir pout \
 --csv-file-pattern news_clean_train.csv \
 --csv-schema-file schema.json \
 --features-file features.json \
 