#!/bin/bash

echo "Installing docker-compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

echo "Building..."
docker-compose -f ./.devcontainer/docker-compose.yml config --services
docker-compose --project-name codestanding_devcontainer -f ./.devcontainer/docker-compose.yml up -d --build
