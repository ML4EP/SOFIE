#!/bin/bash
# Source this file to make a local SOFIE install visible at runtime:
#   source scripts/setup.sh [install-prefix]     (default: <repo>/install)
_sofie_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_sofie_prefix="${1:-$_sofie_root/install}"
export LD_LIBRARY_PATH="$_sofie_prefix/lib:${LD_LIBRARY_PATH:-}"
unset _sofie_root _sofie_prefix
