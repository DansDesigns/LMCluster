#!/bin/sh
# Thin wrapper. All the logic lives in install.py so Linux, macOS and
# Windows share one installer and one set of behaviour.
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 1; }
exec python3 install.py "$@"
