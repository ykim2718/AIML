# Plugin Setup In The Web Interface
Rev. 12 | Created: 2026-08-04 | Updated: 2026-08-31 23:08 CDT

이 문서는 Claude Code 의 web interface 전용이다. 그 환경은 세션마다 container 를 새로 받으므로 설치가 남지 않는다. 설치가 disk 에 남는 desktop interface 는 한 번 설치하고 나면 여기의 절차가 필요 없다.

새로 만들어진 container 가 remote 에서 plugin 을 내려받아 첫 세션부터 규칙을 싣게 하는 방법을 정리한다. 절차와 실패 원인은 모두 Claude Code 2.1.221 에서 실행하여 확인했으며, 확인 과정은 7 장에 남긴다.

## 1. Mechanism

Claude Code 는 세션을 시작할 때 repo 의 `.claude/settings.json` 을 읽어, `extraKnownMarketplaces` 에 적힌 catalog 를 내려받고 `enabledPlugins` 에 적힌 plugin 을 설치한다. container 안에 아무것도 남아 있지 않아도 매번 remote 에서 받아오므로, container 사이에 상태를 옮길 필요가 없다.

이 동작에는 두 가지 전제가 있다.

- 해당 folder 가 trust 된 상태여야 한다. trust 가 없으면 설치 단계 전체를 건너뛴다.
- catalog 와 plugin 의 source 를 모두 network 로 읽을 수 있어야 한다. 읽기에 인증이 필요한 source 라면 git 이 그 인증을 얻을 수 있어야 한다.

두 번째 전제가 이 setup 의 실패 지점이다. catalog 가 private repo 에 있으므로, 그것을 clone 하는 첫 단계에서 git 이 인증을 얻지 못해 아래 오류로 끝나고, plugin 설치는 시작되지도 않는다.

```
Failed to add marketplace: Failed to clone marketplace repository: HTTPS authentication failed.
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

## 2. Catalog And Settings

Table 1. Files this setup requires

| Path | Role |
|---|---|
| `.claude-plugin/marketplace.json` | plugin 의 이름과 위치를 적은 catalog 이며, `ykim2718/Claude-Configuration` 에 있다. |
| `.claude/settings.json` | catalog 를 등록하고 plugin 을 활성화하며, plugin 을 쓰는 repo 마다 있다. |

Catalog 는 plugin 과 같은 repo 에 있으므로, source 를 그 repo 안의 상대 경로로 적는다.

```json
{
  "name": "claude-configuration",
  "owner": {
    "name": "yRocket",
    "email": "ykim2718@gmail.com"
  },
  "description": "yRocket 공용 Claude Code 설정 marketplace (skill/hook 공유).",
  "plugins": [
    {
      "name": "yrocket-rules",
      "source": "./plugins/yrocket-plugins",
      "description": "코드/문서 작성 공용 규칙 skill(coding_rules, md_rules), WordPress post 변환 skill(wp-post-to-github), 대화/필수-로딩 hook."
    }
  ]
}
```

`.claude/settings.json` 의 `enabledPlugins` key 는 `<PLUGIN_NAME>@<MARKETPLACE_NAME>` 형식이며 두 이름 모두 catalog 에 적은 값과 같아야 한다.

```json
{
  "extraKnownMarketplaces": {
    "claude-configuration": {
      "source": {
        "source": "github",
        "repo": "ykim2718/Claude-Configuration"
      }
    }
  },
  "enabledPlugins": {
    "yrocket-rules@claude-configuration": true
  }
}
```

## 3. Credential For The Private Repository

git 에 인증을 주는 방법은 global URL rewrite 이다. rewrite 는 remote 주소 자체에 자격 증명을 넣으므로, credential helper 를 쓰지 않는 경로에서도 적용된다.

```bash
# the trailing form matters: git takes the user name from the URL and still asks for a password
git config --global url."https://x-access-token:<TOKEN>@github.com/".insteadOf "https://github.com/"
```

`<TOKEN>` 은 plugin 이 담긴 repo 를 읽을 수 있는 access token 이다. rewrite 는 그 repo 로만 좁혀 적는 편이 안전하다. host 만 적으면 그 machine 의 모든 github.com 통신에 적용되어 평소 자격 증명을 덮어쓴다.

```bash
# scope the rewrite to one repository
git config --global \
  url."https://x-access-token:<TOKEN>@github.com/ykim2718/Claude-Configuration.git".insteadOf \
  "https://github.com/ykim2718/Claude-Configuration.git"
```

`https://<TOKEN>@github.com/` 처럼 사용자 이름 자리에만 token 을 넣으면 부족하다. git 이 그 값을 사용자 이름으로 받고 password 를 따로 묻기 때문이다. `x-access-token:` 을 앞에 두어야 한 번에 끝난다.

## 4. Container Setup

위 rewrite 와 설치를 container 가 만들어질 때 실행한다. 아래 script 는 Claude Code 가 뜨기 전에 돌아야 하며, 각 줄에 `|| true` 를 붙여 일시적인 실패가 세션 시작을 막지 않게 한다.

```bash
#!/usr/bin/env bash
# container setup: fetch the plugin from the remote repository
set -uo pipefail

git config --global \
  url."https://x-access-token:${CLAUDE_CONFIGURATION_REPO_TOKEN}@github.com/ykim2718/Claude-Configuration.git".insteadOf \
  "https://github.com/ykim2718/Claude-Configuration.git" || true

claude plugin marketplace add ykim2718/Claude-Configuration || true
claude plugin install yrocket-rules@claude-configuration || true
```

`CLAUDE_CONFIGURATION_REPO_TOKEN` 은 container 의 환경 변수로 넘긴다. 이 값은 그 환경을 쓰는 사람이 모두 읽을 수 있으므로, 읽기 권한만 가진 token 을 쓴다.

쓰기 권한은 주지 않는다. 이 token 이 쓰이는 자리는 세션이 시작되기 전 hook 이 marketplace repo 를 clone 하는 한 곳뿐이고, 세션이 뜬 뒤의 push 는 그 세션에 붙은 repo 에 대해 git proxy 가 따로 인증하기 때문이다. Push 에 쓰이지 않는 권한을 담아 두면, 값이 새는 순간 그 repo 를 고칠 수 있는 권한까지 함께 샌다.

이름은 그 token 이 여는 repo 를 그대로 담는다. `GH_TOKEN` 으로 두지 않는 까닭은 어떤 container 가 그 이름을 이미 다른 도구에 쓰고 있고, 그 값은 GitHub token 이 아니므로 rewrite 에 넣으면 멀쩡히 동작하던 자격 증명을 못 쓰는 값으로 덮어쓰기 때문이다.

## 5. Prompt For A New Repository

다른 repo 에 이 setup 을 심을 때는 zip 을 올리고 아래를 그대로 지시한다. 설치를 그 자리에서 실행시키지 않는 것이 요점이다. Plugin 은 세션이 시작될 때 실려서, 지금 설치해도 이 세션에는 잡히지 않는다.

```
첨부 zip 의 `.claude/settings.json` 과 `.claude/hooks/session-start.sh` 를
이 repo 의 `.claude/` 에 넣고, 이 repo 의 origin 기본 branch 에 commit·push 할 것.
기존 `settings.json` 이 있으면 덮어쓰지 말고 병합할 것.
설치는 다음 세션에서 `session-start.sh` 가 하므로 지금 실행하지 말 것.
전제: 이 repo 를 여는 environment 의 설정에 환경 변수 `CLAUDE_CONFIGURATION_REPO_TOKEN` 이
`ykim2718/Claude-Configuration` 을 읽을 수 있는 read-only PAT 로 지정되어 있어야 한다.
없으면 clone 이 인증에서 실패한다.
```

Token 을 넣는 자리는 web 의 environment 설정이며, 그 environment 로 열리는 세션의 container 마다 값이 실린다. Repo 에는 두지 않는다. 한 번 commit 되면 이력에 남아, 뒤늦게 지워도 그 commit 을 아는 사람은 계속 읽을 수 있기 때문이다. 값을 넣는 것은 사람이 하는 일이며, 세션 안의 Claude 는 자기 환경 변수를 바꿀 수 없다.

### 5.1. Setting The Token

값을 넣는 절차는 아래와 같다. Environment 를 여는 별도의 설정 page 나 직접 URL 은 없고, 선택기를 거쳐야 한다.

1. claude.ai/code 에서 message box 바로 위 줄의 cloud icon 을 누른다. 그 icon 에는 지금 쓰는 environment 이름이 적혀 있다.
2. 목록에서 이 repo 를 여는 environment 에 마우스를 올리고, 오른쪽에 나타나는 설정 icon 을 누른다.
3. 열린 dialog 의 **Environment variables** 칸에 `.env` 형식으로 한 줄을 적는다.
4. **Save changes** 를 누른다.

```
CLAUDE_CONFIGURATION_REPO_TOKEN=<READ_ONLY_PAT>
```

값은 세션이 시작할 때 한 번 복사되므로, 지금 돌고 있는 세션은 그대로이고 다음 세션부터 실린다. 그리고 그 environment 를 쓰는 사람은 누구나 이 값을 읽을 수 있으므로, `ykim2718/Claude-Configuration` 을 읽는 것 말고는 아무 권한도 없는 token 을 쓴다.

## 6. Verification

새 세션에서 `/md_rules` 를 호출한다.

```bash
# run from the repository root
claude -p "/md_rules" < /dev/null
```

```
**English:** "Which markdown file should I apply the documentation rules to?"

어떤 .md 파일에 이 규칙을 적용할지 알려주세요.
```

첫 줄이 영어 문장이면 plugin 의 UserPromptSubmit hook 까지 정상이다. 그 hook 이 대화 규칙을 주입하고, 규칙 중 하나가 매 질문을 영어로 옮겨 먼저 보이게 한다.

Skill 의 정식 이름에는 plugin 이름이 namespace 로 붙어 `yrocket-rules:md_rules` 가 된다. 이 namespace 는 catalog 의 entry 이름이 아니라 plugin 자신의 `plugin.json` 에 적힌 이름에서 온다. 이름이 겹치지 않으면 `/md_rules` 처럼 짧게 불러도 같은 skill 이 실린다.

## 7. Experiment Record

빈 HOME 을 만들어 새 container 를 흉내내고, 조건을 하나씩 바꾸며 확인한 결과이다. 이 기록은 catalog 가 public repo 에 있고 plugin 의 source 만 private 이던 때의 것이므로, 다섯째 줄에서 catalog 가 붙는다. Catalog 까지 private 으로 옮긴 지금은 그 줄에서도 인증이 먼저 걸리며, 원인과 해법은 같다.

Table 2. What each condition produced

| Condition | Result |
|---|---|
| 빈 HOME, repo 의 두 파일만 있음 | Unknown command |
| `CLAUDE_CODE_SYNC_PLUGIN_INSTALL` 을 켬 | Unknown command |
| trust 를 미리 승인해 둠 | Unknown command |
| `CLAUDE_CODE_REMOTE` 을 켬 | Unknown command |
| 대화형으로 시작함 | catalog 는 붙고 plugin 은 설치 실패 |
| 설치를 직접 실행하여 오류를 확인함 | git 이 인증을 얻지 못함 |
| URL rewrite 를 걸고 설치함 | 설치 성공 |
| 빈 HOME 에서 rewrite, catalog, 설치를 차례로 실행한 뒤 새 세션 | **정상 로드** |
| `https://<TOKEN>@github.com/` 형태로 rewrite | git 이 password 를 따로 물어 실패 |

앞의 네 줄은 모두 같은 이유로 실패한다. `-p` 로 시작하는 세션은 설치 단계를 아예 실행하지 않으므로, 이 방식으로는 설치 여부를 확인할 수 없다. 다섯째 줄에서 대화형으로 바꾸자 설치 단계가 돌기 시작했고, 그때 비로소 진짜 원인인 인증 실패가 드러났다.

## Appendix A. Terminology

- **catalog**: plugin 의 이름과 위치를 나열한 `marketplace.json` 파일이다.
- **container**: 세션이 실행되는 격리된 실행 환경이다.
- **credential helper**: git 이 remote 인증 정보를 얻을 때 호출하는 외부 program 이다.
- **environment**: web interface 가 세션을 여는 설정 묶음이며, 어떤 repo 를 붙일지와 어떤 환경 변수를 container 에 넘길지를 담는다.
- **git proxy**: 세션의 git 통신이 지나는 중계이며, 그 세션에 붙은 repo 에 대해 container 밖의 자격 증명으로 대신 인증한다.
- **hook**: 정해진 시점에 Claude Code 가 실행하는 command 이다. UserPromptSubmit hook 의 출력은 prompt 마다 context 에 주입된다.
- **marketplace**: catalog 를 통해 plugin 을 배포하는 단위이다.
- **namespace**: skill 이름 앞에 붙어 소속 plugin 을 나타내는 접두사이다.
- **PAT**: personal access token 이며, GitHub 이 발급하는 개인용 접근 자격 증명이다.
- **plugin**: skill, hook 등을 묶어 배포하는 단위이다.
- **skill**: `SKILL.md` 한 개로 정의하는 지시문 묶음이다.
- **trust**: 그 folder 의 설정을 실행해도 되는지에 대한 승인이다.
- **URL rewrite**: git 이 특정 주소를 다른 주소로 바꿔 접속하게 하는 설정이다.

## Appendix B. Bootstrap ZIP

다른 repo 에 이 setup 을 심을 때 올리는 zip 이다. 대상 repo 의 root 에서 풀면 두 파일이 제자리에 놓인다. Code block 의 파일명은 zip 안의 경로이며, 그 repo 에서도 같은 자리에 놓인다.

```text
.claude/
├── settings.json
└── hooks/
    └── session-start.sh
```

Fig 1. Folder structure of the bootstrap zip

이 zip 의 `settings.json` 은 이 repo 의 것과 다르다. 여기에는 marketplace 등록, plugin 활성화, hook 등록만 담고, 이 repo 에만 해당하는 time zone 과 plugin 갱신 hook 은 뺐다.

```json
// .claude/settings.json
{
  "extraKnownMarketplaces": {
    "claude-configuration": {
      "source": {
        "source": "github",
        "repo": "ykim2718/Claude-Configuration"
      }
    }
  },
  "enabledPlugins": {
    "yrocket-rules@claude-configuration": true
  },
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh",
            "timeout": 180
          }
        ]
      }
    ]
  }
}
```

```bash
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
```
