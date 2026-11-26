#!/bin/bash
# install.sh DIR: offline install of this bundle into DIR/venv (run from the unpacked bundle, DIR/app)
set -eu
DIR="$1"; APP="$(cd "$(dirname "$0")" && pwd)"; VENV="$DIR/venv"
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true
rm -rf "$VENV"
"$APP/python/bin/python3.11" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --no-index --no-deps --find-links "$APP/wheels" -r "$APP/requirements.lock"
"$VENV/bin/python" -c 'import mlx.core as mx, mlx_lm, omlx; print("installed: mlx", mx.__version__, "| python", __import__("sys").version.split()[0])'
