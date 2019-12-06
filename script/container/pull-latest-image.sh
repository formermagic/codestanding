#!/bin/bash

# assuming we're running on a CoS instance
docker-credential-gcr configure-docker
# otherwise use the line below
# yes | gcloud auth configure-docker

docker pull gcr.io/codestanding/codestanding-ml:latest