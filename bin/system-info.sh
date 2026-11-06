#!/usr/bin/env bash
set -euo pipefail

repo_root=${BUILD_WORKSPACE_DIRECTORY:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
output="$repo_root/.agents/SYSTEM_INFORMATION.md"

gpu_lines() {
  if ! command -v nvidia-smi >/dev/null; then
    echo "- GPU: none found (nvidia-smi missing)"
    return
  fi
  nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader \
    | awk -F', ' '{print "- GPU: " $1 ", " $2 " VRAM, compute capability " $3 ", driver " $4}'
  nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | sed 's/^/- Driver supports /'
}

mkdir -p "$(dirname "$output")"
{
  echo "# System information"
  echo
  echo "Written by bin/system-info.sh on $(date -I). Rerun it after hardware changes."
  echo
  echo "- OS: $(. /etc/os-release && echo "$PRETTY_NAME"), kernel $(uname -r)"
  echo "- CPU: $(lscpu | sed -n 's/^Model name: *//p'), $(nproc) threads"
  echo "- RAM: $(awk '/^MemTotal/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo)"
  gpu_lines
  echo "- Free disk at the repo: $(df -h --output=avail "$repo_root" | tail -1 | tr -d ' ')"
} > "$output"
echo "Wrote $output"
