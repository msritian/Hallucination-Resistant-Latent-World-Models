#!/bin/bash
# Run on your Mac from the repo root: bundles the code that jobs need into cluster/code.tar.gz
set -e
cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf cluster/code.tar.gz \
    --exclude='__pycache__' --exclude='.git' \
    src third_party/tdmpc2/tdmpc2
ls -lh cluster/code.tar.gz
