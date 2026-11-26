#!/usr/bin/env bash
# Build the MDZ file front end (mdz_file) against SZ3, which ships MDZ in tools/mdz (header only).
# Usage: md_compress/ext/build_mdz.sh [PREFIX]   (default PREFIX=$HOME/.local/mdpack; binary: PREFIX/bin/mdz_file)
# Needs: git, cmake >= 3.19, a C++17 compiler, libzstd (headers + library). Then: export MDPACK_MDZ=PREFIX/bin/mdz_file
set -euo pipefail
PREFIX="${1:-$HOME/.local/mdpack}"; SZ3_COMMIT="${SZ3_COMMIT:-28387bee81d54f00d4644b1e3f52424b1dcafcc1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git clone -q https://github.com/szcompressor/SZ3.git "$TMP/SZ3" && git -C "$TMP/SZ3" checkout -q "$SZ3_COMMIT"
cmake -S "$TMP/SZ3" -B "$TMP/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_MDZ=ON -DCMAKE_INSTALL_PREFIX="$PREFIX" > /dev/null
cmake --build "$TMP/build" -j > /dev/null && cmake --install "$TMP/build" > /dev/null
g++ -O3 -std=c++17 -I"$PREFIX/include" -I"$PREFIX/include/MDZ" "$HERE/mdz_file.cpp" -o "$PREFIX/bin/mdz_file" -lzstd
echo "built $PREFIX/bin/mdz_file (SZ3 $SZ3_COMMIT); also installed: $PREFIX/bin/mdz (the original in-memory tool)"
