#!/usr/bin/env bash
# Compiles shaders, then builds/runs the given bin.
# Usage: ./run.sh <bin> [args...]
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <bin> [args...]" >&2
    exit 1
fi

bin="$1"
shift

cargo run -p compiler
cargo run -p "$bin" -- "$@"
