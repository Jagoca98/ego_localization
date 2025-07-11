#!/bin/bash

# This script is used to run ego_localization playground

# Get the current directory
set -e
CURRENT_DIR=$(pwd)

#  Load the environment variables from the .env file
set -o allexport
source .env
set -o allexport

# Ensure environment variables are set
: "${USER_ID:?Need to set USER_ID}"
: "${USER_NAME:?Need to set USER_NAME}"
: "${GROUP_ID:?Need to set GROUP_ID}"
: "${GROUP_NAME:?Need to set GROUP_NAME}"
: "${WORKSPACE:?Need to set WORKSPACE}"
: "${DOCKER_IMAGE_NAME:?Need to set DOCKER_IMAGE_NAME}"

# Generating the data inside a docker container
echo "Holiwilis to the ego_localization container"
if docker run \
        --name $DOCKER_IMAGE_NAME  \
        --network host \
        --device /dev/dri:/dev/dri \
        -v $CURRENT_DIR/data/:/data:ro \
        -v $CURRENT_DIR/data/output:/data/output:rw \
        -v $CURRENT_DIR/catkin_ws:/catkin_ws/ \
        -u $USER_ID:$GROUP_ID \
        -e DISPLAY=$DISPLAY \
	    -e TERM=xterm-256color \
        --rm \
        -it \
        $DOCKER_IMAGE_NAME \
        bash; then
    echo "Chauchau OkiDoki"
else
    echo "Chauchau"
    exit 1
fi
