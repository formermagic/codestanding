#!/bin/bash

if [ $# -ne 2 ]; then
    echo "You must pass 2 arguments: SRC_FILE and DEST"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

BUCKET=gs://diff-dataset-bucket
SRC_FILE=$BUCKET/$1
DEST=$2

$DIR/setup-config.sh

gsutil cp $SRC_FILE $DEST

$DIR/clear-config.sh