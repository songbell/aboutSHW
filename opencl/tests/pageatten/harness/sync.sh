#!/bin/bash
# Push the harness (and optionally a kernel under test) into the only directory the `llm`
# container can see.
#
# The container mounts /home/intel/ceciliapeng -> /ceciliapeng and nothing else, so scripts
# living in this repo are invisible to it. It also has its *own* copy of the kernel tree at
# /ceciliapeng/bell/aboutSHW, separate from the one you edit here. Measuring an edit therefore
# always means: edit here -> sync -> run in the container.
#
#   ./sync.sh                       # harness only
#   ./sync.sh path/to/kernel.cm     # harness + drop a kernel into the probe dir
#
# Then, inside the container:
#   docker exec llm bash -lc 'cd /ceciliapeng/bell/aboutSHW/opencl/tests/pageatten && \
#       python /ceciliapeng/kernel_harness/bench_paired.py'
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST=/home/intel/ceciliapeng/kernel_harness
PROBE=/home/intel/ceciliapeng/kernel_probe_src

mkdir -p "$DST" "$PROBE"
cp "$SRC"/*.py "$DST"/
echo "harness -> $DST"

if [[ $# -ge 1 ]]; then
    cp "$1" "$PROBE"/
    echo "kernel  -> $PROBE/$(basename "$1")"
    echo
    echo "NOTE: a kernel copied here is a snapshot. Re-run sync.sh after every edit, or you"
    echo "      will be measuring the previous version -- this has happened."
fi

if command -v docker >/dev/null && docker ps --format '{{.Names}}' | grep -qx llm; then
    busy=$(docker exec llm bash -lc \
        "ps -eo args | grep -iE 'benchmark|visual_language|pytest' | grep -v grep | head -3" || true)
    if [[ -n "$busy" ]]; then
        echo
        echo "WARNING: the container is already running GPU work -- measurements will be junk:"
        echo "$busy"
    fi
fi
