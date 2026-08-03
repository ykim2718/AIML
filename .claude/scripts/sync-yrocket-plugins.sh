#!/usr/bin/env bash
__version__="0.2.2.2026.8.3"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)
# Refresh this repository's copy of the yrocket skills and prompt rules.
#
# The rules are vendored under .claude/ rather than installed as a plugin, because a
# plugin only loads in a session that already had it installed at startup, which a
# fresh container never does. Their source of truth stays in the yrocket-plugins
# plugin of ykim2718/Claude-Configuration, so this script copies from there.
#
# It never commits; review the diff it reports and commit it yourself.
#
# Usage:
#   .claude/scripts/sync-yrocket-plugins.sh [SOURCE_FOLDER]
#
#   SOURCE_FOLDER  an existing checkout of ykim2718/Claude-Configuration. Omit it to
#                  clone the repository into a temporary folder.

set -euo pipefail

source_url="https://github.com/ykim2718/Claude-Configuration.git"
plugin_path="plugins/yrocket-plugins"
skill_names="md_rules coding_rules"
rule_names="conversation_rules.md skill_loading_rules.md"

script_folder="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_folder="$(cd "$script_folder/../.." && pwd)"

usage() {
  cat <<USAGE
Refresh .claude/skills and .claude/hooks from the yrocket-plugins plugin in $source_url.

Usage:
  $(basename "${BASH_SOURCE[0]}") [SOURCE_FOLDER]

  SOURCE_FOLDER  an existing checkout of ykim2718/Claude-Configuration. Omit it to
                 clone the repository into a temporary folder.

The script never commits; review the diff it reports and commit it yourself.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ $# -gt 1 ]; then
  echo "sync: expected at most one SOURCE_FOLDER, got $#" >&2
  usage >&2
  exit 2
fi

cleanup_folder=""
cleanup() {
  # An unconditional && here would leave the trap returning 1 and become the
  # script's exit status on a successful run.
  if [ -n "$cleanup_folder" ]; then
    rm -rf "$cleanup_folder"
  fi
}
trap cleanup EXIT

if [ $# -eq 1 ]; then
  source_folder="$1"
else
  cleanup_folder="$(mktemp -d)"
  source_folder="$cleanup_folder/claude-configuration"
  echo "sync: cloning $source_url"
  # A private source repository has to be reachable from this machine or session;
  # the clone fails loudly here rather than leaving the vendored copy stale.
  git clone --depth 1 "$source_url" "$source_folder"
fi

source_plugin="$source_folder/$plugin_path"
if [ ! -d "$source_plugin" ]; then
  echo "sync: $source_plugin not found" >&2
  exit 1
fi

# Every source file is checked before anything is overwritten, so a source that
# renamed or dropped a file cannot leave this repository half updated.
for skill in $skill_names; do
  if [ ! -f "$source_plugin/skills/$skill/SKILL.md" ]; then
    echo "sync: $source_plugin/skills/$skill/SKILL.md not found" >&2
    exit 1
  fi
done
for rule in $rule_names; do
  if [ ! -f "$source_plugin/hooks/$rule" ]; then
    echo "sync: $source_plugin/hooks/$rule not found" >&2
    exit 1
  fi
done

for skill in $skill_names; do
  rm -rf "${project_folder:?}/.claude/skills/$skill"
  mkdir -p "$project_folder/.claude/skills"
  cp -a "$source_plugin/skills/$skill" "$project_folder/.claude/skills/$skill"
done
for rule in $rule_names; do
  cp -a "$source_plugin/hooks/$rule" "$project_folder/.claude/hooks/$rule"
done

echo "sync: vendored skills and prompt rules from $source_folder"
targets=".claude/skills .claude/hooks"
if git -C "$project_folder" diff --quiet -- $targets &&
  [ -z "$(git -C "$project_folder" ls-files --others --exclude-standard -- $targets)" ]; then
  echo "sync: no change"
else
  git -C "$project_folder" status --short -- $targets
  echo "sync: review the diff above and commit it"
fi
