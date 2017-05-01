#! /bin/bash

rm -fr tout && python -m trainer.task \
 --train-data-paths=run_new_sd/exout/features_train-* \
 --eval-data-paths=run_new_sd/exout/features_test-* \
 --job-dir tout \
 --analysis-output-dir run_new_sd/pout \
 --model-type linear_classification \
 --top-n 20 \
 --max-steps 5000 \
 --train-batch-size 10 \
 --eval-batch-size 10 \

