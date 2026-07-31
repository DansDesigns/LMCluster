#!/bin/sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || { echo "run ./install.sh first"; exit 1; }
. .venv/bin/activate
exec python -m lmcluster "${1:-lmcluster.toml}"
