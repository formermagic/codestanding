#!/bin/bash
python /workspace/src/tokenizer.py --diff_dir /workspace/tmp --output /workspace/tmp/result --max_workers 20
zip -r /workspace/tmp/tokenized.zip /workspace/tmp/result
gsutil cp /workspace/tmp/tokenized.zip gs://diff-dataset-bucket/tokenized_diff_dataset.zip