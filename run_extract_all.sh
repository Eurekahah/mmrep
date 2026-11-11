#!/usr/bin/env sh
# Once all checkpoints are available, run all 128 files. Pass --setting
# pretrained to run the 32 current baseline files instead.
set -eu
case "$0" in
    */*) cd "${0%/*}" ;;
esac
if [ -n "${MMREP_PYTHON:-}" ]; then
    python_bin="$MMREP_PYTHON"
elif [ -x .venv/bin/python ]; then
    python_bin=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
    python_bin=.venv/Scripts/python.exe
else
    python_bin=python
fi
exec "$python_bin" -m extract.extract_features --model all --setting all \
    --dataset all --split all --cond complete "$@"
