#!/bin/bash

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

checkpoint_dir='exp/hkust'
COMPUTE_ID='20180313'

./bin/run-cluster.sh 1:1:16 \
  --train_files bin/hkust/hkust_phn_train.csv \
  --dev_files bin/hkust/hkust_phn_dev.csv \
  --test_files bin/hkust/hkust_phn_dev.csv \
  --alphabet_config_path bin/hkust/alphabet.txt \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --n_hidden 600 \
  --epoch 40 \
  --checkpoint_dir "$checkpoint_dir" \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --display_step 1 \
  --validation_step 1 \
  --log_level 1 \
  --log_traffic False \
  "$@"
