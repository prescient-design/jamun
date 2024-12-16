#!/usr/bin/env bash

uv export \
  --frozen \
  --no-emit-project \
  --no-hashes \
  --output-file requirements.txt
