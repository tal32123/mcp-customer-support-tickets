#!/usr/bin/env bash
# Rewrite the messages of placeholder ("x" / "j") commits in this repo's
# history to follow Conventional Commits. Safe to re-run: if a SHA is no
# longer in history (e.g. already rewritten), it is silently ignored.
#
# Requires: bash, git. Force-push is left to the operator.
#
# Usage:
#   scripts/rename-commits.sh           # rewrite history locally
#   DRY_RUN=1 scripts/rename-commits.sh # list affected commits, no rewrite
#
# After running, push with:
#   git push --force-with-lease origin main

set -euo pipefail

# Disable autocrlf for every git call in this script so a Windows
# core.autocrlf=true setting cannot produce phantom "modified" entries
# when stash/checkout touches files. Encoded for GIT_CONFIG_PARAMETERS.
export GIT_CONFIG_PARAMETERS="'core.autocrlf=false'"

# sha -> conventional message. Extend this map as new placeholder commits appear.
declare -A MESSAGES=(
  [5ba5bcc32308156f5e52ae1bd35b5aa512eee66f]="chore: add .gitignore and initial design doc"
  [d06c5149e40e3308978c266289826dc7c6cbdc9e]="docs: tweak design HTML files"
  [29f8877c6e6c24dc6fa1096f7a49b0eaba47e5e2]="docs: remove design preview artifacts"
  [abf884f43d75473bf1ee11d4c6c5da0ce78dbb15]="docs: refresh one-page design and add architecture diagram"
  [6d7e15ac11b530209b757e44016e4823383f19ce]="docs: add one-page design and PDF export"
  [ee74b9940393061ee6b346cfbb483a69155c46a0]="test(draft-reply): add prompt tests and tweak schema resource"
  [6fc0d3cd7bbd56440efd9f6610cfe2ec9434514b]="chore: add LICENSE and remove rerank stub"
  [cb7156f0f7cd23bda103cfe4d3fb2f50176a0a08]="refactor: remove unused LLM client adapters"
  [7bb1498fd35671c6c517a86e9f9f0f304ce5a239]="refactor(draft-reply): rework prompt and trim config/server"
  [04d12e53de7e71a583b9c92f66039970f256fc17]="test(eval): add RAG evaluation suite (known-item, facets, metrics)"
)

cd "$(git rev-parse --show-toplevel)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Would rewrite (those present in current history):"
  for sha in "${!MESSAGES[@]}"; do
    if git cat-file -e "${sha}^{commit}" 2>/dev/null; then
      printf '  %s  %s\n' "${sha:0:7}" "${MESSAGES[$sha]}"
    fi
  done
  exit 0
fi

backup="backup-before-rename-$(date +%Y%m%d-%H%M%S)"
git branch "$backup"
echo "Backup branch created: $backup"

stashed=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  git stash push -u -m "pre-rename-stash"
  stashed=1
  echo "Working tree stashed."
fi

# Stash + checkout can leave EOL-only phantom diffs on Windows; clear them
# so filter-branch's "no unstaged changes" precondition holds.
git checkout -- .

# Build a POSIX `case` block from the associative array so filter-branch's
# msg-filter (which runs under /bin/sh) can dispatch by $GIT_COMMIT.
filter=""
for sha in "${!MESSAGES[@]}"; do
  msg=${MESSAGES[$sha]//\"/\\\"}
  filter+="    $sha) echo \"$msg\" ;;"$'\n'
done

cleanup() {
  if (( stashed )); then
    git stash pop || echo "WARNING: stash pop failed; recover with 'git stash list'."
  fi
}
trap cleanup EXIT

FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --msg-filter "
case \"\$GIT_COMMIT\" in
$filter    *) cat ;;
esac
" -- --all

echo
echo "Done. Verify with: git log --oneline"
echo "If happy, push with: git push --force-with-lease origin main"
echo "To roll back:        git reset --hard $backup"
