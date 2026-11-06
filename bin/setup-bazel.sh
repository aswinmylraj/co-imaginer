#!/usr/bin/env bash
set -euo pipefail

readonly bazelisk_version=v1.30.0
readonly bazelisk_sha256=a248ae2b2b62b5baebb748f6fba7203bff1a27e5d01b55399a8406744cb539e8
readonly install_dir="$HOME/.local/bin"

repo_root=${BUILD_WORKSPACE_DIRECTORY:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd "$repo_root"

install_bazelisk() {
  local bazel_path="$install_dir/bazel"
  if [[ -x "$bazel_path" ]] && echo "$bazelisk_sha256  $bazel_path" | sha256sum --check --status; then
    return
  fi
  mkdir -p "$install_dir"
  local download
  download=$(mktemp)
  curl -fsSL -o "$download" "https://github.com/bazelbuild/bazelisk/releases/download/$bazelisk_version/bazelisk-linux-amd64"
  if ! echo "$bazelisk_sha256  $download" | sha256sum --check --status; then
    rm -f "$download"
    echo "setup-bazel: bazelisk checksum mismatch" >&2
    exit 1
  fi
  install -m 0755 "$download" "$bazel_path"
  rm -f "$download"
}

fetch_tools() {
  bazel build @gitleaks//:bin @ruff//:bin @uv//:bin
}

sync_python_env() {
  "$repo_root/bin/uv" sync --locked
  "$repo_root/bin/uv" cache prune
}

export PATH="$install_dir:$PATH"
install_bazelisk
fetch_tools
sync_python_env
echo "Bazel setup complete: $(bazel --version)"
