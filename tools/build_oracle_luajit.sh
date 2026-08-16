#!/usr/bin/env bash
# Builds the RNG-oracle LuaJIT into tools/oracle/luajit.
#
# Balatro (built Feb 2025) bundles a LuaJIT 2.1 of that era. The system
# luajit CANNOT be used as an oracle for math.random: distro rolling builds
# (e.g. CachyOS 2.1.1780076327+b925b3e) have diverging math.randomseed
# behavior. pseudohash/pseudoseed are pure f64 arithmetic and are
# version-independent, but regenerate ALL vectors with this pinned build:
#
#   tools/oracle/luajit tools/gen_rng_vectors.lua > sim/core/tests/data/rng_vectors.tsv
set -euo pipefail
cd "$(dirname "$0")"
PIN=8358eb0cce  # v2.1 tip of 2025-01-13, contemporaneous with the game build
rm -rf oracle-build
git clone -q https://github.com/LuaJIT/LuaJIT.git oracle-build
git -C oracle-build checkout -q "$PIN"
make -C oracle-build -j"$(nproc)" -s
mkdir -p oracle
cp oracle-build/src/luajit oracle/luajit
rm -rf oracle-build
./oracle/luajit -v
