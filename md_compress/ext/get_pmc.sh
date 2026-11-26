#!/usr/bin/env bash
# Fetch PMC 1.0, the authors' command line tool for Dvorak, Manak and Vasa, "Predictive compression of molecular
# dynamics trajectories", J Mol Graph Model 96 (2020) 107531 (https://jandvorak-uwb.github.io/pmc/), and a .NET runtime.
# PMC targets .NET Core 2.1; it runs on .NET 8 with DOTNET_ROLL_FORWARD=Major (set by md_compress).
# Usage: md_compress/ext/get_pmc.sh [PREFIX]   (default PREFIX=$HOME/.local/mdpack)
# Then: export MDPACK_PMC=PREFIX/pmc/bin/pmc.dll MDPACK_DOTNET=PREFIX/dotnet/dotnet
set -euo pipefail
PREFIX="${1:-$HOME/.local/mdpack}"; mkdir -p "$PREFIX/pmc"; TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -sSL -o "$TMP/pmc.zip" https://github.com/jandvorak-uwb/pmc/releases/download/1.0/pmc.zip
echo "531d7bc3dba6921773a41b78c4f6107c5c0d745766fae7ef9b45983545421478  $TMP/pmc.zip" | sha256sum -c - > /dev/null
unzip -q -o "$TMP/pmc.zip" -d "$PREFIX/pmc"
if ! command -v dotnet > /dev/null && [ ! -x "$PREFIX/dotnet/dotnet" ]; then
  curl -sSL -o "$TMP/dotnet-install.sh" https://dot.net/v1/dotnet-install.sh
  bash "$TMP/dotnet-install.sh" --runtime dotnet --channel 8.0 --install-dir "$PREFIX/dotnet" > /dev/null
fi
echo "PMC in $PREFIX/pmc/bin/pmc.dll; dotnet: $(command -v dotnet || echo "$PREFIX/dotnet/dotnet")"
