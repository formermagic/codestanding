#!/bin/bash

docker-compose -f ./.devcontainer/docker-compose.yml config --services

docker-compose --project-name codestanding_devcontainer -f ./.devcontainer/docker-compose.yml up -d --build
