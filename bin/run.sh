#!/bin/bash

export DOCKER_CLI_HINTS=false

if [ "$(docker ps -q --filter ancestor=daltunay)" ]; then
    docker stop $(docker ps -q --filter ancestor=daltunay)
    docker rm $(docker ps -q --filter ancestor=daltunay)
fi

docker build -t daltunay . &&
docker run -p 8501:8501 daltunay
