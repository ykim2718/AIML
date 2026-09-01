#!/bin/bash
# .claude/hooks/session-start.sh
# Install the yrocket-rules plugin so its skills load in a web session.
#
# .claude/settings.json names the marketplace and enables the plugin, but a
# declaration is not an install. A web session gets a fresh container whose
# plugin store is empty, and nothing there installs a GitHub marketplace
# plugin. The plugin carries its own SessionStart hook, but that one only
# updates plugins that are already installed and exits when there are none, so
# it cannot bootstrap itself. The bootstrap has to sit outside the plugin, and
# this is it.
#
# Local machines keep their own install across runs, so this only does anything
# in the throwaway container.
set -uo pipefail

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
command -v claude >/dev/null 2>&1 || exit 0

MARKETPLACE='ykim2718/Claude-Configuration'
PLUGIN='yrocket-rules@claude-configuration'
REPO_URL='https://github.com/ykim2718/Claude-Configuration.git'
LOG="$HOME/.claude/plugin-bootstrap.log"

mkdir -p "$(dirname "$LOG")"
{
  date '+=== session start %Y-%m-%d %H:%M:%S'
  # The marketplace repository is private, so git needs a credential before the
  # clone. CLAUDE_CONFIGURATION_REPO_TOKEN is a read-only token for that one
  # repository, injected as a container environment variable and never committed
  # here. The name says which repository it opens, and it is not GH_TOKEN: that
  # name is already taken by other tooling in some containers, whose value is
  # not a GitHub token, and writing it into the rewrite replaces a working
  # credential with one that cannot authenticate. The rewrite is scoped to this
  # one URL, because a host-wide rewrite would override the credential every
  # other github.com operation uses. Without the token the clone still runs and
  # fails on authentication, which the log then names.
  if [ -n "${CLAUDE_CONFIGURATION_REPO_TOKEN:-}" ]; then
    git config --global \
      url."https://x-access-token:${CLAUDE_CONFIGURATION_REPO_TOKEN}@github.com/ykim2718/Claude-Configuration.git".insteadOf \
      "$REPO_URL" || echo 'credential rewrite FAILED'
  else
    echo 'CLAUDE_CONFIGURATION_REPO_TOKEN not set; the clone will fail unless the session already carries a credential'
  fi
  # Both are idempotent: re-adding a marketplace and re-installing a plugin
  # that are already present succeed and change nothing.
  claude plugin marketplace add "$MARKETPLACE" || echo 'marketplace add FAILED'
  claude plugin install "$PLUGIN" || echo 'install FAILED'
  claude plugin list
} >>"$LOG" 2>&1

# A missing plugin is worth a log line, never a failed session start.
exit 0
