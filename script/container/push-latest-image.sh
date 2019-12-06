#!/bin/bash

if [ $# -ne 1 ]; then
    echo "You must pass a tag"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TAG=$1

$DIR/../gcloud/setup-config.sh

docker tag codestanding_devcontainer_codestanding-ml:latest gcr.io/codestanding/codestanding-ml:$TAG
docker push gcr.io/codestanding/codestanding-ml:$TAG
gcloud container images list-tags gcr.io/codestanding/codestanding-ml

$DIR/../gcloud/clear-config.sh