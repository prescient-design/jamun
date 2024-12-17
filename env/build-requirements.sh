#!/usr/bin/env bash

uv pip compile \
  --verbose \
  --no-emit-index-url \
  --emit-find-links \
  --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
  --only-binary mdtraj \
  -c env/constraints.txt \
  -o env/requirements.txt \
  requirements.in &&

  uv pip compile \
    --verbose \
    --no-emit-index-url \
    --emit-find-links \
    --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
    --only-binary mdtraj \
    -c env/constraints-analysis.txt \
    -o env/requirements-analysis.txt \
    requirements-analysis.in &&
  
  uv pip compile \
  --verbose \
  --no-emit-index-url \
  --emit-find-links \
  --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
  --only-binary mdtraj \
  -c env/constraints.txt \
  -o env/requirements-dev.txt \
  requirements.in \
  requirements-dev.in
