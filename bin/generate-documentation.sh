#!/usr/bin/env bash
set -euo pipefail

repo_root=${BUILD_WORKSPACE_DIRECTORY:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
cd "$repo_root"

llm="${DOCUMENTATION_LLM:-claude}"
today=$(date -I)
output="documentation/progress-$today.tex"

fail() {
  echo "generate-documentation: $1" >&2
  exit 1
}

update_main() {
  git diff --quiet && git diff --cached --quiet || fail "uncommitted changes. Commit or stash them first."
  git switch --quiet main
  git pull --ff-only --quiet origin main
}

previous_commit() {
  local previous_file
  previous_file=$(find documentation -maxdepth 1 -name 'progress-*.tex' ! -name "progress-$today.tex" | sort | tail -n 1)
  [[ -n "$previous_file" ]] || return 0
  local commit
  commit=$(sed -n 's/^\\newcommand{\\progresscommit}{\([0-9a-f]\{40\}\)}$/\1/p' "$previous_file")
  [[ -n "$commit" ]] || fail "no \\progresscommit found in $previous_file"
  git cat-file -e "$commit^{commit}" 2>/dev/null || fail "commit $commit from $previous_file is not in this repository"
  echo "$commit"
}

changes_prompt() {
  local base=$1
  cat <<EOF
Write the body of the "Changes" section of a LaTeX progress report for this repository.
Follow the writing rules in AGENTS.md, included below.
Output only LaTeX body text: \subsection*{} groups (Documents, Code, Experiments) with itemize lists.
Cite short commit hashes. Describe only the changes in the diff below. No preamble, no markdown fences.

== AGENTS.md
$(cat AGENTS.md)

== commits
$(git log --no-merges --date=short --format='%h %ad %an: %s' "${base:+$base..}HEAD")

== diff
$(git diff "${base:-$(git hash-object -t tree /dev/null)}" HEAD -- . ':(exclude)uv.lock' ':(exclude)documentation/progress-*.tex')
EOF
}

run_llm() {
  case "$llm" in
    claude) claude -p ;;
    ollama:*) ollama run "${llm#ollama:}" ;;
    *) fail "DOCUMENTATION_LLM must be claude or ollama:<model>, got '$llm'" ;;
  esac
}

write_report() {
  local base=$1 changes=$2
  cat > "$output" <<EOF
\documentclass{article}
\usepackage[hidelinks]{hyperref}
\newcommand{\progresscommit}{$(git rev-parse HEAD)}

\title{Progress report}
\author{}
\date{$today}

\begin{document}
\maketitle

\section*{Commit hash}
Main at \texttt{\progresscommit}. Previous report: \texttt{${base:-none}}.

\section*{Changes}
$changes

\section*{Benchmarks}
No benchmark results yet.

\end{document}
EOF
}

build_pdfs() {
  local tex
  for tex in documentation/*.tex; do
    latexmk -pdf -interaction=nonstopmode -quiet -outdir=build "$tex"
    cp "build/$(basename "${tex%.tex}").pdf" documentation/
  done
}

update_main
base=$(previous_commit)
changes=$(changes_prompt "$base" | run_llm | sed '/^```/d')
[[ -n "$changes" ]] || fail "the LLM returned no text"
write_report "$base" "$changes"
build_pdfs
echo "Wrote $output and its PDF. It is uncommitted on main."
