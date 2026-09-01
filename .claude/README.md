# Claude Rules Setup
Rev. 6 | Created: 2026-08-03 | Updated: 2026-08-31 22:12 CDT

이 폴더는 세션이 시작될 때마다 공용 규칙이 실리도록 구성되어 있다. 규칙의 실체는 `ykim2718/Claude-Configuration` 의 `yrocket-rules` plugin 이고, 그 repo 가 marketplace catalog 도 함께 담는다. 이 repo 는 그 plugin 을 켜는 설정과 그것을 설치하는 hook 을 가지며, 규칙의 사본도 아직 남아 있다.

## 1. Layout

Table 1. Files and their roles

| Path | Role |
|---|---|
| `.claude/settings.json` | marketplace 를 등록하고 plugin 을 활성화하며, hook 을 걸고, container 의 time zone 을 저자의 지역 시간대로 맞춘다. |
| `.claude/hooks/session-start.sh` | 새 container 에서 marketplace 를 붙이고 plugin 을 설치한다. |
| `.claude/hooks/update-plugins.sh` | 이미 설치된 plugin 을 세션 시작 때 갱신한다. |
| `.claude/hooks/conversation_rules.md` | 대화 규칙의 사본이며, 지금은 어떤 hook 도 읽지 않는다. |
| `.claude/hooks/skill_loading_rules.md` | 필수 skill 로딩 규칙의 사본이며, 마찬가지로 읽는 곳이 없다. |
| `.claude/skills/` | plugin 이 담은 skill 의 사본이다. |

Catalog 는 이 repo 에 없다. `.claude-plugin/marketplace.json` 은 plugin 과 같은 repo 인 `ykim2718/Claude-Configuration` 의 최상위에 있으며, Claude Code 가 그 경로에서 catalog 를 찾는다.

## 2. Settings

`.claude/settings.json` 의 `extraKnownMarketplaces` 는 catalog 를 어디서 읽을지 정하고, `enabledPlugins` 는 그 catalog 의 어떤 plugin 을 켤지 정한다. 두 값이 있어야 사용자가 `/plugin marketplace add` 를 직접 실행하지 않아도 새 세션에서 규칙이 실린다.

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

`enabledPlugins` 의 key 는 `<PLUGIN_NAME>@<MARKETPLACE_NAME>` 형식이며, 두 이름 모두 catalog 에 적은 값과 같아야 한다.

## 3. Session Start Hook

선언은 설치가 아니다. Web 세션은 설치된 plugin 이 하나도 없는 container 를 받으므로, `enabledPlugins` 만으로는 plugin 이 실리지 않는다. `session-start.sh` 가 그 자리를 메워 `claude plugin marketplace add` 와 `claude plugin install` 을 실행한다. 둘 다 idempotent 이므로 이미 있는 것을 다시 실행해도 달라지는 것이 없다.

`Claude-Configuration` 이 private 이므로 그 clone 에는 자격 증명이 필요하다. 읽기 권한만 가진 token 을 `PLUGIN_REPO_TOKEN` 환경 변수로 container 에 넘기면, hook 이 그 값으로 URL rewrite 를 걸고 clone 한다. Token 이 없으면 clone 이 인증에서 실패하고 plugin 은 실리지 않는다.

## 4. Setup For Another Repository

대상 repo 의 root 에서 `.claude/settings.json` 과 `.claude/hooks/session-start.sh` 를 두면 끝난다. Catalog 는 `Claude-Configuration` 에 있으므로 대상 repo 에 만들지 않는다.

```bash
# run from the target repository root
mkdir -p .claude/hooks
cp <SOURCE_REPO>/.claude/settings.json .claude/settings.json
cp <SOURCE_REPO>/.claude/hooks/session-start.sh .claude/hooks/session-start.sh
```

commit 하고 push 한 뒤 새 세션을 열어 `/md_rules` 가 호출되면 plugin 이 실린 것이다.

## 5. Update

plugin 의 내용이 바뀌면 `/plugin marketplace update` 로 catalog 를 갱신한다. Catalog 와 plugin 이 같은 repo 에 있으므로 별도의 version 표기가 없으며, 그 repo 의 새 commit 이 곧 새 version 이 된다.

## 6. Caution

대화 규칙을 주입하던 UserPromptSubmit hook 은 이 repo 의 설정에서 지웠다. Plugin 이 같은 일을 하는 자기 hook 을 갖고 있어, plugin 이 실리는 세션에서 규칙이 두 번 주입되었기 때문이다.

남은 사본은 `.claude/hooks` 의 규칙 문서 두 개와 `.claude/skills` 이다. 읽는 hook 이 없어진 앞의 둘은 이제 아무 데도 쓰이지 않고, `.claude/skills` 는 plugin 이 실리면 같은 skill 을 두 벌로 만든다. 지울지 남길지는 아직 정해져 있지 않다.

## Appendix A. Terminology

- **catalog**: plugin 의 이름과 위치를 나열한 `marketplace.json` 파일이다.
- **hook**: 정해진 시점에 Claude Code 가 실행하는 command 이다. UserPromptSubmit hook 의 출력은 prompt 마다 context 에 주입된다.
- **idempotent**: 여러 번 실행해도 한 번 실행한 것과 결과가 같은 성질이다.
- **marketplace**: catalog 를 통해 plugin 을 배포하는 단위이다.
- **plugin**: skill, hook 등을 묶어 배포하는 단위이며, 세션 시작 시점에 설치되어 있어야 실린다.
- **skill**: `SKILL.md` 한 개로 정의하는 지시문 묶음이다. `/<name>` 으로 직접 호출하거나 Claude 가 필요할 때 스스로 로드한다.
