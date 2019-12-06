#!/bin/bash

FILE=~/.docker/config.json
if [ -f $FILE ]; then
    rm -rf $FILE
fi
