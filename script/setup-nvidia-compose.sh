#!/bin/bash

DAEMON_PATH=/etc/docker/daemon.json
CONTENTS="""\
{
    \"default-runtime\":\"nvidia\",
    \"runtimes\": {
        \"nvidia\": {
            \"path\": \"nvidia-container-runtime\",
            \"runtimeArgs\": []
        }
    }
}\
"""

if [ -f $DAEMON_PATH ]; then
    echo "$CONTENTS" > $DAEMON_PATH
fi



