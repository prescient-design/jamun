#!/usr/bin/env bash

uv export \
  --frozen \
  --no-emit-project \
  --no-hashes \
  --output-file env/requirements.txt

uv pip compile \
    --verbose \
    -o env/requirements-analysis.txt \
    requirements-analysis.in
