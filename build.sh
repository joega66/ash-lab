#!/usr/bin/env bash
# Builds, then compiles shaders.
# Usage: ./build.sh
set -euo pipefail

cargo build
cargo run --bin compiler