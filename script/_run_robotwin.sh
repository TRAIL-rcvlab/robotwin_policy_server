#!/bin/bash
set -eo pipefail

IMAGE="robotwin:cu121-py310"
POLICY_NAME="$1"
CONTAINER_NAME="robotwin_${POLICY_NAME}"   # 修改容器名字

ASSETS_PATH="/data2/blzou/dataset/robotwin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(dirname "$SCRIPT_DIR")"

# 检查镜像是否存在
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "❌ Image: $IMAGE is not exist."
    exit 1
fi

#!/bin/bash
set -eo pipefail

IMAGE="robotwin:cu121-py310"
POLICY_NAME="$1"
CONTAINER_NAME="robotwin_${POLICY_NAME}"

ASSETS_PATH="/data2/blzou/dataset/robotwin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(dirname "$SCRIPT_DIR")"

# 检查镜像是否存在
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "❌ Image: $IMAGE does not exist."
    exit 1
fi

# 检查容器是否存在
if docker container inspect "$CONTAINER_NAME" &>/dev/null; then
    # 容器存在，检查是否在运行
    STATUS=$(docker container inspect "$CONTAINER_NAME" --format='{{.State.Running}}')
    if [ "$STATUS" != "true" ]; then
        echo "🔄 Container '$CONTAINER_NAME' exists but is not running. Restarting..."
        docker start "$CONTAINER_NAME"
        echo "✅ Container restarted."
    else
        echo "✅ Container '$CONTAINER_NAME' is already running."
    fi
else
    # 容器不存在，创建并启动
    echo "🆕 Create new container '$CONTAINER_NAME' "

    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --shm-size=8g \
        --network=host \
        --privileged \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -v "$CURRENT_DIR":/workspace \
        -v "$ASSETS_PATH/assets:/workspace/assets" \
        -v "$ASSETS_PATH/data:/workspace/data" \
        -v "$ASSETS_PATH/data_real:/workspace/data_real" \
        -v "$ASSETS_PATH/ckpt:/workspace/ckpt" \
        -w /workspace \
        "$IMAGE" \
        bash -c "
            tail -f /dev/null
        "

    echo "✅ Container created and patched, waiting for startup..."
fi
sleep 2

# 进入容器
echo "🚪 Entering container $CONTAINER_NAME ..."
docker exec -it "$CONTAINER_NAME" bash