# Plugin Setup And Verification
rev. 1

이 문서는 `yrocket-plugins` plugin 으로 공용 규칙을 싣는 절차와, Claude Code 2.1.221 에서 실제로 실행하여 확인한 결과를 정리한다. plugin 의 실체는 `ykim2718/Claude-Configuration` 의 `plugins/yrocket-plugins` folder 에 있고, 이 repo 는 그 위치를 가리키는 catalog 와 설정만 가진다.

## 1. Files

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

`.claude/settings.json` 의 `extraKnownMarketplaces` 는 catalog 를 어디서 읽을지 정하고, `enabledPlugins` 는 그 catalog 의 어떤 plugin 을 켤지 정한다. key 는 `<PLUGIN_NAME>@<MARKETPLACE_NAME>` 형식이며 두 이름 모두 catalog 에 적은 값과 같아야 한다.

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

## 2. Installation

위 두 파일만으로는 설치가 일어나지 않는다. 설치를 한 번 실행해야 하며, 아래 두 command 가 그 역할을 한다. 인자는 catalog 를 담은 repo 와 `<PLUGIN_NAME>@<MARKETPLACE_NAME>` 이다.

```bash
# run once per machine
claude plugin marketplace add ykim2718/AIML
claude plugin install yrocket-plugins@yrocket-marketplace
```

`Claude-Configuration` 은 private 이므로 이 machine 에 git credential helper 가 설정되어 있어야 두 번째 command 가 성공한다. 설치 결과는 `claude plugin list` 로 확인한다.

```
Installed plugins:

  > yrocket-plugins@yrocket-marketplace
    Version: 5afc699f89c8-5953cdc5
    Scope: user
    Status: enabled
```

## 3. Verification

설치 이후 새로 시작한 세션에서 `/md_rules` 를 호출하면 skill 이 실린다.

```bash
# run from the repository root
claude -p "/md_rules" < /dev/null
```

```
**English:** "What markdown document task would you like me to perform?"

md_rules skill을 로드했습니다. 어떤 .md 파일을 작성/수정/검토할지 알려주세요.
```

첫 줄이 영어 문장으로 시작하면 plugin 의 UserPromptSubmit hook 까지 정상이다. 그 hook 이 대화 규칙을 주입하고, 규칙 중 하나가 매 질문을 영어로 옮겨 먼저 보이는 것이다.

Skill 의 정식 이름에는 plugin 이름이 namespace 로 붙어 `yrocket-rules:md_rules` 가 된다. 이 namespace 는 catalog 의 entry 이름이 아니라 plugin 자신의 `plugin.json` 에 적힌 이름에서 온다. 이름이 겹치지 않으면 `/md_rules` 처럼 짧게 불러도 같은 skill 이 실린다.

## 4. Timing Constraint

plugin 은 세션이 시작되는 시점에 이미 설치되어 있어야 실린다. 아래는 확인한 결과이다.

Table 2. What each state produces in a newly started session

| State at session start | `/md_rules` |
|---|---|
| 두 파일만 있고 설치 이력이 없음 | Unknown command |
| 두 파일을 user settings 에 두고 설치 이력이 없음 | Unknown command |
| SessionStart hook 이 그 세션에서 설치를 실행함 | Unknown command |
| 이전 세션까지 설치가 끝나 있음 | 정상 로드 |

세 번째 행이 핵심이다. SessionStart hook 이 설치를 마쳐도 그 세션은 이미 plugin 목록을 확정한 뒤이므로 규칙이 실리지 않고, 다음 세션부터 실린다. 따라서 세션마다 새 container 를 만드는 환경에서는 다음 세션이 오지 않으므로 이 방식으로 규칙이 실리지 않는다.

그런 환경에서 첫 세션부터 규칙이 필요하면 plugin 대신 skill 과 hook 파일을 repo 의 `.claude/` 에 직접 두어야 한다. checkout 에 이미 들어 있는 파일은 설치 단계 없이 읽히기 때문이다.

## 5. Setup For Another Repository

대상 repo 의 root 에서 `.claude/settings.json` 만 복사하면 된다. catalog 는 이 repo 에 둔 것을 그대로 가리키므로 대상 repo 에 다시 만들 필요가 없다.

```bash
# run from the target repository root
mkdir -p .claude
cp <SOURCE_REPO>/.claude/settings.json .claude/settings.json
```

commit 하고 push 한 뒤, 그 machine 에서 2 장의 command 를 한 번 실행하고 새 세션을 열어 3 장의 방법으로 확인한다.

## 6. Update

원본 repo 의 규칙이 바뀌면 아래로 갱신한다. `ref` 가 branch 이므로 별도의 version 표기 없이 새 commit 이 곧 새 version 이 된다. 갱신 역시 다음 세션부터 반영된다.

```bash
claude plugin marketplace update yrocket-marketplace
claude plugin update yrocket-plugins@yrocket-marketplace
```

## Appendix A. Terminology

- **catalog**: plugin 의 이름과 위치를 나열한 `marketplace.json` 파일이다.
- **container**: 세션이 실행되는 격리된 실행 환경이다.
- **credential helper**: git 이 remote 인증 정보를 얻을 때 호출하는 외부 program 이다.
- **hook**: 정해진 시점에 Claude Code 가 실행하는 command 이다. UserPromptSubmit hook 의 출력은 prompt 마다 context 에 주입되고, SessionStart hook 은 세션이 시작될 때 한 번 실행된다.
- **marketplace**: catalog 를 통해 plugin 을 배포하는 단위이다.
- **namespace**: skill 이름 앞에 붙어 소속 plugin 을 나타내는 접두사이다.
- **plugin**: skill, hook 등을 묶어 배포하는 단위이다.
- **skill**: `SKILL.md` 한 개로 정의하는 지시문 묶음이다.
- **sparse clone**: repo 의 일부 folder 만 내려받는 clone 방식이다.
