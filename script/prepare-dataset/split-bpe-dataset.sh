#!/bin/bash

# TODO:
# 1) Create a tmp dir for train/valid/test splits
# 2) Create train/valid/test splits from full train files
# 3) Replace full train files with train/valid/test splits

echo "Creating tmp dir..."

DATASET_PREFIX=/workspace/tmp/dataset_200k.bpe/train
TMP_DIR=/workspace/tmp/dataset_200k.bpe/split.tmp

if [ ! -d $TMP_DIR ]; then
    mkdir $TMP_DIR
fi

echo "Splitting dataset into tmp dir..."

python -m src.split_dataset split \
            --dataset_prefix=$DATASET_PREFIX \
            --exts='.diff, .msg' \
            --split-ratio='0.8, 0.15, 0.05' \
            --dest-path=$TMP_DIR


echo "Moving splits into root dir..."

rm -f $DATASET_PREFIX{.*}
mv $TMP_DIR/* $TMP_DIR/..
rm -rf $TMP_DIR