#!/bin/bash
set -e 

cd $(dirname "$0")

# format cpp code
find . -not -path "./bazel-*" -type f \( -iname "*.cc" -o -iname "*.c" -o -iname "*.cpp" -o -iname "*.h" -o -iname "*.hpp" \) | \
    xargs clang-format -i --sort-includes

# format python
find . -type f -name '*py' \( -not -path "*/__pycache__/*" -and -not -path "*/.git/*" -and -not -path "*/ipynb_checkpoints/*" \) -prune \
    -exec yapf -i {} +
