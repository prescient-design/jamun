#!/usr/bin/env bash

uv pip compile \
  --verbose \
  --no-emit-index-url \
  --emit-find-links \
  --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
  -c constraints.txt \
  -o requirements.txt \
  ../requirements.in
