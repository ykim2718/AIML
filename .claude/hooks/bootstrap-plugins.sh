#!/usr/bin/env bash
__version__="0.1.0.2026.8.3"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)
# Register this repository's plugin marketplace and install the plugins that
# .claude/settings.json enables.
#
# Claude Code web sessions start from a fresh container and do not act on
# enabledPlugins on their own, so the repository has to bootstrap itself. The
# plugin's own update hook cannot do this because it only runs once the plugin
# is already installed.
#
# The marketplace is registered from the local checkout, so this step needs no
# network access. Resolving a plugin body that lives in another repository still
# does, and that repository must be reachable from the session.

set -u

script_folder="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_folder="${CLAUDE_PROJECT_DIR:-$(cd "$script_folder/../.." && pwd)}"
settings="$project_folder/.claude/settings.json"
manifest="$project_folder/.claude-plugin/marketplace.json"
log="$HOME/.claude/plugin-bootstrap.log"

mkdir -p "$(dirname "$log")"
exec 3>>"$log"
echo "=== $(date "+%Y-%m-%d %H:%M:%S") bootstrap in $project_folder" >&3

# report writes to stdout, which Claude Code injects into the session context.
report() {
  echo "plugin bootstrap: $*"
  echo "plugin bootstrap: $*" >&3
}

# report_error writes to stderr, which stays visible even though a failed
# bootstrap exits non-zero and its stdout is dropped.
report_error() {
  echo "plugin bootstrap: $*" >&2
  echo "plugin bootstrap: $*" >&3
}

if ! command -v claude >/dev/null 2>&1; then
  report "claude CLI not found; skipping"
  exit 0
fi

if [ ! -f "$settings" ]; then
  report "$settings not found; skipping"
  exit 0
fi

# Enabled plugins are the "<plugin>@<marketplace>": true keys of settings.json.
enabled_plugins=$(
  grep -oE '"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+"[[:space:]]*:[[:space:]]*true' "$settings" |
    grep -oE '[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+' | sort -u
)
if [ -z "$enabled_plugins" ]; then
  report "no enabled plugins in settings.json; nothing to do"
  exit 0
fi

installed_plugins=$(claude plugin list 2>&3)
configured_marketplaces=$(claude plugin marketplace list 2>&3)

# Register the marketplace this repository ships, from the local checkout.
if [ -f "$manifest" ]; then
  local_marketplace=$(grep -oE '"name"[[:space:]]*:[[:space:]]*"[^"]+"' "$manifest" | head -1 |
    sed -E 's/.*"name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
  if [ -z "$local_marketplace" ]; then
    report_error "cannot read marketplace name from $manifest"
  elif echo "$configured_marketplaces" | grep -qE "(^|[^A-Za-z0-9_.-])$local_marketplace([^A-Za-z0-9_.-]|$)"; then
    echo "marketplace $local_marketplace already configured" >&3
  elif claude plugin marketplace add "$project_folder" >&3 2>&3; then
    report "registered marketplace $local_marketplace from $project_folder"
    configured_marketplaces=$(claude plugin marketplace list 2>&3)
  else
    report_error "FAILED to register marketplace $local_marketplace; see $log"
  fi
fi

installed_count=0
failed_count=0
for plugin in $enabled_plugins; do
  if echo "$installed_plugins" | grep -qF "$plugin"; then
    echo "$plugin already installed" >&3
    continue
  fi
  marketplace="${plugin##*@}"
  if ! echo "$configured_marketplaces" | grep -qE "(^|[^A-Za-z0-9_.-])$marketplace([^A-Za-z0-9_.-]|$)"; then
    report_error "FAILED $plugin: marketplace $marketplace is not configured"
    failed_count=$((failed_count + 1))
    continue
  fi
  if claude plugin install "$plugin" >&3 2>&3; then
    installed_count=$((installed_count + 1))
  else
    # A plugin body hosted in another repository fails here when that
    # repository is out of the session's reach.
    report_error "FAILED to install $plugin; see $log"
    failed_count=$((failed_count + 1))
  fi
done

if [ "$installed_count" -gt 0 ]; then
  report "installed $installed_count plugin(s); restart the session if their skills and hooks are not active yet"
fi
if [ "$failed_count" -gt 0 ]; then
  exit 1
fi
exit 0
