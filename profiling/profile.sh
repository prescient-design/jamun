#!/usr/bin/env bash

export PATH=$PATH:/usr/local/cuda/bin

nsys profile \
  -w true \
  --wait=all \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu  \
  -x true \
  -o profiling/nsys.profile \
  --force-overwrite true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage true \
  --cudabacktrace=all \
  "$@"