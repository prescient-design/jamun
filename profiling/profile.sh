#!/usr/bin/env bash

export PATH=$PATH:/usr/local/cuda/bin
export HYDRA_FULL_ERROR=1
export TORCH_COMPILE_DEBUG=1
export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE=1
export TORCH_LOGS="recompiles"

nsys profile \
  -w true \
  --wait=all \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu  \
  -x true \
  -o nsys.profile \
  --force-overwrite true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage true \
  --cudabacktrace=all \
  "$@"