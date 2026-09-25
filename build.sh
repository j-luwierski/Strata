#!/bin/sh
# build.sh - rebuild the Strata engine and install it as engine/strata.
#
# Run this after changing any source file or after switching branches.  When it finishes, restart
# the server (run-iq3_xxs.sh) so it picks up the new binary.
#
#   ./build.sh                    build the engine (+ strata-vision when its sources changed)
#   ./build.sh --clean            start from a fresh build directory (slow, fixes broken builds)
set -e
cd "$(dirname "$0")"

PYTHON=.venv/bin/python
[ -x "$PYTHON" ] || PYTHON=python3

if [ "$1" = "--clean" ]; then
    rm -rf build
fi

# configure once; the CUDA architecture 89 covers RTX 40/50 (change for another GPU)
cmake -B build -S . -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target strata -j

cp build/strata engine/strata
echo "engine installed: engine/strata ($(date '+%H:%M:%S'))"
echo "restart the server (run-iq3_xxs.sh) to use it"
