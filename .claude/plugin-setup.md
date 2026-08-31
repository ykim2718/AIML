# Plugin Setup And Verification
Rev. 4 | Created: 2026-08-04 | Updated: 2026-08-31 18:23 CDT

이 문서는 새로 만들어진 container 가 remote 에서 plugin 을 내려받아 첫 세션부터 규칙을 싣게 하는 방법을 정리한다. 절차와 실패 원인은 모두 Claude Code 2.1.221 에서 실행하여 확인했으며, 확인 과정은 6 장에 남긴다.

## 1. Mechanism

Claude Code 는 세션을 시작할 때 repo 의 `.claude/settings.json` 을 읽어, `extraKnownMarketplaces` 에 적힌 catalog 를 내려받고 `enabledPlugins` 에 적힌 plugin 을 설치한다. container 안에 아무것도 남아 있지 않아도 매번 remote 에서 받아오므로, container 사이에 상태를 옮길 필요가 없다.

이 동작에는 두 가지 전제가 있다.

- 해당 folder 가 trust 된 상태여야 한다. trust 가 없으면 설치 단계 전체를 건너뛴다.
- catalog 와 plugin 의 source 를 모두 network 로 읽을 수 있어야 한다. 읽기에 인증이 필요한 source 라면 git 이 그 인증을 얻을 수 있어야 한다.

두 번째 전제가 이 setup 의 실패 지점이었다. catalog 는 붙지만 plugin 의 source 를 clone 하는 단계에서 git 이 인증을 얻지 못해 아래 오류로 끝난다.

```
Failed to clone repository for git-subdir source:
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

## 2. Files In The Repository

Table 1. Files this setup requires

| Path | Role |
|---|---|
| `.claude-plugin/marketplace.json` | plugin 의 이름과 위치를 적은 catalog 이다. |
| `.claude/settings.json` | catalog 를 등록하고 plugin 을 활성화한다. |

`.claude-plugin/marketplace.json` 은 `git-subdir` source 로 원본 repo 의 한 folder 만 sparse clone 한다.

```json
{
  "name": "yrocket-marketplace",
  "owner": {
    "name": "yrocket",
    "email": "ykim2718@gmail.com"
  },
  "plugins": [
    {
      "name": "yrocket-plugins",
      "source": {
        "source": "git-subdir",
        "url": "https://github.com/ykim2718/Claude-Configuration.git",
        "path": "plugins/yrocket-plugins",
        "ref": "main"
      }
    }
  ]
}
```

`.claude/settings.json` 의 `enabledPlugins` key 는 `<PLUGIN_NAME>@<MARKETPLACE_NAME>` 형식이며 두 이름 모두 catalog 에 적은 값과 같아야 한다.

```json
{
  "extraKnownMarketplaces": {
    "yrocket-marketplace": {
      "source": {
        "source": "github",
        "repo": "ykim2718/AIML"
      }
    }
  },
  "enabledPlugins": {
    "yrocket-plugins@yrocket-marketplace": true
  }
}
```

## 3. Credential For The Plugin Source

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
  url."https://x-access-token:${PLUGIN_REPO_TOKEN}@github.com/ykim2718/Claude-Configuration.git".insteadOf \
  "https://github.com/ykim2718/Claude-Configuration.git" || true

claude plugin marketplace add ykim2718/AIML || true
claude plugin install yrocket-plugins@yrocket-marketplace || true
```

`PLUGIN_REPO_TOKEN` 은 container 의 환경 변수로 넘긴다. 이 값은 그 환경을 쓰는 사람이 모두 읽을 수 있으므로, 읽기 권한만 가진 token 을 쓴다.

이름을 `GH_TOKEN` 으로 두지 않는다. 어떤 container 는 그 이름을 이미 다른 도구가 쓰고 있고, 그 값은 GitHub token 이 아니므로 rewrite 에 넣으면 멀쩡히 동작하던 자격 증명을 못 쓰는 값으로 덮어쓴다.

## 5. Verification

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

## 6. Experiment Record

빈 HOME 을 만들어 새 container 를 흉내내고, 조건을 하나씩 바꾸며 확인한 결과이다.

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
- **hook**: 정해진 시점에 Claude Code 가 실행하는 command 이다. UserPromptSubmit hook 의 출력은 prompt 마다 context 에 주입된다.
- **marketplace**: catalog 를 통해 plugin 을 배포하는 단위이다.
- **namespace**: skill 이름 앞에 붙어 소속 plugin 을 나타내는 접두사이다.
- **plugin**: skill, hook 등을 묶어 배포하는 단위이다.
- **skill**: `SKILL.md` 한 개로 정의하는 지시문 묶음이다.
- **sparse clone**: repo 의 일부 folder 만 내려받는 clone 방식이다.
- **trust**: 그 folder 의 설정을 실행해도 되는지에 대한 승인이다.
- **URL rewrite**: git 이 특정 주소를 다른 주소로 바꿔 접속하게 하는 설정이다.
