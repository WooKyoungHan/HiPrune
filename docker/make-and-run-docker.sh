# DO NOT TOUCH THIS FILE.

# Usage (at the root of the repository):
 export MODEL_DIR=/home/cjss7894/ckpts
 export DATASET_DIR=/home/cjss7894/Datasets
#  $ ./docker/make-and-run-docker.sh


BASE_IMAGE_WITH_TAG=nvcr.io/nvidia/pytorch:25.11-py3

# Build the base image
# docker build \
#   -t ${BASE_IMAGE_WITH_TAG} \
#   -f docker/Dockerfile .

# Build the user image
docker build \
  --build-arg BASE_IMAGE_WITH_TAG=${BASE_IMAGE_WITH_TAG} \
  --build-arg USER_ID=$(id -u) \
  --build-arg USER_NAME=$(id -un) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg GROUP_NAME=$(id -gn) \
  -t ${BASE_IMAGE_WITH_TAG}-$(whoami) \
  -f docker/Dockerfile.user .

# Run the container
docker run --rm -it --gpus all \
  --name HiPrune-$(whoami) --net=host \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_DATASETS_CACHE=/datasets/.cache/hf \
  -v "$(pwd)":/workspace \
  -v "${MODEL_DIR}":/models \
  -v "${DATASET_DIR}":/datasets \
  ${BASE_IMAGE_WITH_TAG}-$(whoami) \
  bash
