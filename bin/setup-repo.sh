#!/usr/bin/env bash
set -euo pipefail

readonly ponytail_commit=96608e078b2d5ea607c5c3b8b5a54d1c1fba6846
readonly apt_packages=(
  curl direnv nodejs
  latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-publishers
)

repo_root=${BUILD_WORKSPACE_DIRECTORY:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd "$repo_root"

install_system_packages() {
  local missing=()
  for package in "${apt_packages[@]}"; do
    dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed" || missing+=("$package")
  done
  if (( ${#missing[@]} )); then
    echo "Installing: ${missing[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing[@]}"
  fi
}

write_envrc() {
  cat > .envrc <<'EOF'
PATH_add "$HOME/.local/bin"
export VIRTUAL_ENV="$PWD/.venv"
PATH_add .venv/bin
PATH_add bin
EOF
  direnv allow .
  local shell_name rc_file
  shell_name=$(basename "$SHELL")
  rc_file="$HOME/.${shell_name}rc"
  if ! grep -qs 'direnv hook' "$rc_file"; then
    echo "eval \"\$(direnv hook $shell_name)\"" >> "$rc_file"
    echo "Added the direnv hook to $rc_file. Open a new shell."
  fi
}

install_git_hooks() {
  local hooks_dir
  hooks_dir=$(git rev-parse --git-path hooks)
  mkdir -p "$hooks_dir"
  ln -sfn "$repo_root/bin/hooks/pre-commit" "$hooks_dir/pre-commit"
  git config commit.gpgsign true
  git config tag.gpgsign true
  git config user.signingkey >/dev/null || echo "WARNING: no user.signingkey set. Commits will fail until signing is configured."
}

install_ponytail_rules() {
  mkdir -p .agents
  curl -fsSL "https://raw.githubusercontent.com/DietrichGebert/ponytail/$ponytail_commit/AGENTS.md" -o .agents/PONYTAIL_RULES.md
  printf '\n## Project rules\n\nAlso follow AGENTS.md in this repository.\n' >> .agents/PONYTAIL_RULES.md
}

install_system_packages
"$repo_root/bin/system-info.sh"
write_envrc
install_git_hooks
install_ponytail_rules
echo "Setup complete."
