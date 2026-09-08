# Plugin Setup In A Web Interface
Rev. 0 | Created: 2026-09-08 | Updated: 2026-09-08 18:53 CDT

Web session 은 plugin 이 하나도 설치되지 않은 새 container 에서 시작한다. 이 문서는 그 container 에 plugin 을 싣는 두 경로 가운데, 저장소의 hook 이 아니라 cloud environment 의 setup script 를 쓰는 쪽을 다룬다.

## 1. Purpose

- **Problem Statement**: `.claude/settings.json` 의 `enabledPlugins` 는 선언일 뿐이어서, 새 container 를 받는 web session 에서는 plugin 이 설치되지 않고 skill 도 실리지 않는다.
- **Goal**: Setup script 하나로 marketplace 와 plugin 을 Claude Code 가 뜨기 전에 설치하여, 세션이 열린 시점에 `claude plugin list` 가 대상 plugin 을 enabled 로 보이게 한다.
- **Non-Goal**: plugin 과 skill 의 작성, catalog 파일의 구성, local 설치는 다루지 않는다.

## 2. Setup Script

아래 script 를 cloud environment 설정의 **Setup script** 칸에 넣으면 설치가 끝난다. Setup script 는 새 세션이 시작될 때 Claude Code 가 뜨기 전에 root 로 실행되는 Bash script 이므로 [[2](#ref-2)], 세션이 첫 prompt 를 받는 시점에는 plugin 이 이미 자리에 있다.

```bash
# Claude Code cloud environment — Setup script field
set -uo pipefail

MARKETPLACE='ykim2718/Claude-Configuration'
REPO_URL='https://github.com/ykim2718/Claude-Configuration.git'
PLUGINS='yrocket-md-doc@claude-configuration yrocket-coding@claude-configuration yrocket-wordpress@claude-configuration'
LOG="$HOME/plugin-setup.log"

{
  date -u '+=== setup %Y-%m-%dT%H:%M:%SZ'
  # The marketplace repository is private, so git needs a credential before the
  # clone. The rewrite is scoped to this one URL, because a host-wide rewrite
  # would override the credential every other github.com operation uses.
  git config --global \
    url."https://x-access-token:<GITHUB_PAT>@github.com/ykim2718/Claude-Configuration.git".insteadOf \
    "$REPO_URL"
  # Both are idempotent: re-adding a marketplace and re-installing a plugin
  # that are already present succeed and change nothing.
  claude plugin marketplace add "$MARKETPLACE"
  claude plugin marketplace update claude-configuration
  for p in $PLUGINS; do
    claude plugin install "$p"
  done
  claude plugin list
} >>"$LOG" 2>&1

# A missing plugin is worth a log line, never a failed session start.
exit 0
```

Setup script 에는 세 가지 제약이 있다 [[2](#ref-2)].

- Script 가 non-zero 로 끝나면 세션이 시작되지 않는다. 마지막 줄의 `exit 0` 이 그 자리이며, 설치가 실패해도 세션은 열린다.
- 전체 실행 시간을 5분 안에 두어야 environment cache 가 만들어진다. 위 script 는 repo 하나를 clone 하는 분량이므로 여유가 있다.
- 설치에는 network 가 필요하다. Network access 를 None 으로 둔 environment 에서는 marketplace clone 이 실패한다.

## 3. Token

`Claude-Configuration` 이 private 이므로 clone 에 자격 증명이 필요하다. Script 의 `<GITHUB_PAT>` 자리에 그 repo 의 contents 읽기 권한만 가진 token 을 넣는다.

- Token 값은 environment 설정 안에만 둔다. 저장소에 commit 하지 않으며, 이 문서에도 자리표시자만 남긴다.
- URL rewrite 는 그 repo 하나에만 건다. `github.com` 전체에 걸면 다른 저장소를 다루는 git 작업의 자격 증명까지 덮는다.
- Token 이 없거나 만료되면 clone 이 인증에서 실패하고 plugin 은 설치되지 않는다. 실패한 사실은 log 에만 남는다.

## 4. Verification

`$HOME/plugin-setup.log` 의 마지막 실행 기록에 대상 plugin 이 모두 enabled 로 나오면 설치된 것이다.

```text
=== setup 2026-09-08T23:21:51Z ===
Adding marketplace…Cloning via HTTPS: https://github.com/ykim2718/Claude-Configuration.git
√ Successfully added marketplace: claude-configuration (declared in user settings)
Updating marketplace: claude-configuration...√ Successfully updated marketplace: claude-configuration
Installing plugin "yrocket-md-doc@claude-configuration"...√ Successfully installed plugin: yrocket-md-doc@claude-configuration (scope: user)
Installed plugins:

  > yrocket-md-doc@claude-configuration
    Version: fb10ecb87d78
    Scope: user
    Status: √ enabled
```

세션 안에서 확인할 때는 `/md_rules` 를 불러 본다. Skill 이 실리면 plugin 이 설치된 것이고, 이름을 찾지 못하면 log 를 본다.

## 5. Comparison

Setup script 와 SessionStart hook 은 실행 시점과 설정 위치가 다르므로, 둘 중 하나를 고르는 대신 함께 두어도 된다. 앞의 것이 설치를 끝내면 뒤의 것은 이미 있는 것을 확인하고 지나간다.

Table 1. Setup script and SessionStart hook

| Aspect | Setup script | SessionStart hook |
| --- | --- | --- |
| Configured in | Cloud environment 설정의 Setup script 칸에 둔다. | 저장소의 `.claude/settings.json` 에 둔다. |
| Runs | 새 세션이 시작될 때 Claude Code 가 뜨기 전에 실행된다. | Claude Code 가 뜬 뒤 세션 시작 시점에 실행된다. |
| Applies to | 그 environment 를 쓰는 모든 저장소에 적용된다. | 그 저장소를 여는 모든 세션에 적용된다. |
| Repeats | Script 나 허용 host 가 바뀔 때, 그리고 cache 가 약 7일 뒤 만료될 때 다시 실행된다. | 세션마다 실행된다. |
| On failure | Non-zero 로 끝나면 세션이 시작되지 않는다. | 세션은 그대로 시작된다. |

Setup script 는 첫 실행 뒤 filesystem snapshot 으로 저장되고, 이후 세션은 그 snapshot 에서 시작하며 script 를 건너뛴다 [[2](#ref-2)]. 설치된 plugin 은 disk 에 남는 것이므로 snapshot 에 함께 실려 다음 세션으로 넘어간다. 이미 있던 세션을 다시 여는 경우에는 script 가 실행되지 않는다.

## 6. Further Work

N/A — 설치 대상 plugin 이 모두 실리고 있어 남은 방향이 없다.

## References

<a id="ref-1"></a>[1] Anthropic. [Use Claude Code on the web](https://code.claude.com/docs/en/claude-code-on-the-web). Claude Code Documentation.<br>
<a id="ref-2"></a>[2] Anthropic. [Configure cloud environments](https://code.claude.com/docs/en/cloud-environments). Claude Code Documentation.

---

## Appendix A. Terminology

- **catalog**: plugin 의 이름과 위치를 나열한 `marketplace.json` 파일이다.
- **cloud environment**: network 허용 범위, 환경 변수, setup script 를 담아 둔 cloud session 설정이며, web 과 terminal 등 모든 경로의 cloud session 이 같은 것을 쓴다 [[1](#ref-1)].
- **hook**: 정해진 시점에 Claude Code 가 실행하는 command 이다.
- **idempotent**: 여러 번 실행해도 한 번 실행한 것과 결과가 같은 성질이다.
- **marketplace**: catalog 를 통해 plugin 을 배포하는 단위이다.
- **PAT (Personal Access Token)**: GitHub 계정을 대신하여 저장소에 접근하는 token 이다.
- **plugin**: skill, hook 등을 묶어 배포하는 단위이며, 세션 시작 시점에 설치되어 있어야 실린다.
- **setup script**: 새 cloud session 이 시작될 때 Claude Code 보다 먼저 root 로 실행되는 Bash script 이다.
- **skill**: `SKILL.md` 한 개로 정의하는 지시문 묶음이다.
