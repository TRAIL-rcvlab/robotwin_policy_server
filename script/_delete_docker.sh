#!/bin/bash
set -e

CONTAINER_NAME="robotwin_dp3"

# 检查容器是否存在
if docker container inspect "$CONTAINER_NAME" &>/dev/null; then
    echo "🗑️  Deleting container '$CONTAINER_NAME' ..."
    docker stop "$CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME"
    echo "✅ Container deleted."
else
    echo "❌ Container '$CONTAINER_NAME' does not exist."
fi