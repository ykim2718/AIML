#!/usr/bin/env bash
__version__="0.1.0.2026.8.3"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)
# Refresh the vendored copy of the yrocket-rules plugin from its source repository.
#
# The plugin body is vendored into this repository so that a session can install it
# without reaching a private repository. Its source of truth stays in
# ykim2718/Claude-Configuration, so this script pulls that repository and overwrites
# the vendored copy. It never commits; review the diff and commit it yourself.
#
# Usage:
#   .claude/scripts/sync-yrocket-rules.sh [SOURCE_FOLDER]
#
#   SOURCE_FOLDER  an existing checkout of ykim2718/Claude-Configuration. Omit it to
#                  clone the repository into a temporary folder.

set -euo pipefail

source_url="https://github.com/ykim2718/Claude-Configuration.git"
plugin_path="plugins/yrocket-rules"

script_folder="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_folder="$(cd "$script_folder/../.." && pwd)"

usage() {
  cat <<USAGE
Refresh the vendored copy of the yrocket-rules plugin from $source_url.

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
  if [ ! -d "$source_folder/$plugin_path" ]; then
    echo "sync: $source_folder/$plugin_path not found" >&2
    exit 1
  fi
else
  cleanup_folder="$(mktemp -d)"
  source_folder="$cleanup_folder/claude-configuration"
  echo "sync: cloning $source_url"
  # A private source repository has to be reachable from this machine or session;
  # the clone fails loudly here rather than leaving the vendored copy stale.
  git clone --depth 1 "$source_url" "$source_folder"
fi

source_plugin="$source_folder/$plugin_path"
target_plugin="$project_folder/$plugin_path"

if [ ! -f "$source_plugin/.claude-plugin/plugin.json" ]; then
  echo "sync: $source_plugin is not a plugin folder" >&2
  exit 1
fi

rm -rf "$target_plugin"
mkdir -p "$(dirname "$target_plugin")"
cp -a "$source_plugin" "$target_plugin"

echo "sync: vendored $plugin_path from $source_folder"
if git -C "$project_folder" diff --quiet -- "$plugin_path" &&
  [ -z "$(git -C "$project_folder" ls-files --others --exclude-standard -- "$plugin_path")" ]; then
  echo "sync: no change"
else
  git -C "$project_folder" status --short -- "$plugin_path"
  echo "sync: review the diff above and commit it"
fi
