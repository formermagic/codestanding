#!/bin/bash

if [ $# -ne 2 ]; then
    echo "You must pass 2 arguments: FILE and DEST"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

BUCKET=gs://diff-dataset-bucket
FILE=$1
DEST=$BUCKET/$2

$DIR/setup-config.sh

gsutil cp $FILE $DEST

$DIR/clear-config.sh