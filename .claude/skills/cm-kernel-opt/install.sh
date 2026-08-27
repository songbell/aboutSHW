#!/bin/bash
# Expose the agents and this skill at the workspace root (/home/intel/bell), which is the
# session cwd but is NOT a git repository -- so the canonical, versioned copies live here in
# aboutSHW and the root only gets symlinks. Re-run after cloning.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # aboutSHW/.claude
ROOT="${1:-/home/intel/bell}/.claude"
mkdir -p "$ROOT/skills"
ln -sfn "$SRC/agents" "$ROOT/agents"
ln -sfn "$SRC/skills/cm-kernel-opt" "$ROOT/skills/cm-kernel-opt"
echo "linked:"; ls -l "$ROOT/agents" "$ROOT/skills/cm-kernel-opt"
