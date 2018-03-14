#!/bin/sh
set -xe
if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -f "bin/hkust/hkust_phn_train.csv" ]; then
    echo "Please prepare hkust data first"
    exit 1
fi;

COMPUTE_KEEP_DIR='exp/hkust'
COMPUTE_ID='20180313'
if [ ! -d "${COMPUTE_KEEP_DIR}" ]; then
    mkdir -p $COMPUTE_KEEP_DIR
fi

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/hkust"))')
fi

python -u DeepSpeech.py \
  --train_files bin/hkust/hkust_phn_train.csv \
  --dev_files bin/hkust/hkust_phn_dev.csv \
  --test_files bin/hkust/hkust_phn_dev.csv \
  --alphabet_config_path bin/hkust/alphabet.txt \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --n_hidden 2048 \
  --epoch 30 \
  --checkpoint_dir "$checkpoint_dir" \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --display_step 1 \
  --validation_step 1 \
  "$@"
