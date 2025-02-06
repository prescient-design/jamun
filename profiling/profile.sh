#!/usr/bin/env bash

export PATH=$PATH:/usr/local/cuda/bin

nsys profile \
  -w true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu  \
  -x true \
  -o nsys.profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage true \
  --cudabacktrace=all \
  "$@"