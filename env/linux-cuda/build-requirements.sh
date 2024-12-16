#!/usr/bin/env bash

uv pip compile \
  --verbose \
  --no-emit-index-url \
  --emit-find-links \
  --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
  --only-binary mdtraj \
  -c env/linux-cuda/constraints.txt \
  -o env/linux-cuda/requirements.txt \
  requirements.in &&

  uv pip compile \
    --verbose \
    --no-emit-index-url \
    --emit-find-links \
    --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
    --only-binary mdtraj \
    -c env/linux-cuda/constraints-analysis.txt \
    -o env/linux-cuda/requirements-analysis.txt \
    requirements.in \
    requirements-analysis.in &&
  
  uv pip compile \
  --verbose \
  --no-emit-index-url \
  --emit-find-links \
  --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
  --only-binary mdtraj \
  -c env/linux-cuda/constraints.txt \
  -o env/linux-cuda/requirements-dev.txt \
  requirements.in \
  requirements-dev.in
