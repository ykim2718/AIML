# Claude Rules Setup
Rev. 4 | Created: 2026-08-03 | Updated: 2026-08-10 22:15 CDT

이 폴더는 세션이 시작될 때마다 공용 규칙이 자동으로 실리도록 구성되어 있다. 규칙의 실체는 `ykim2718/Claude-Configuration` 의 `yrocket-plugins` plugin 이고, 이 repo 는 그 plugin 을 가리키는 marketplace catalog 와 설정만 가진다.

## 1. Layout

Table 1. Files and their roles

| Path | Role |
|---|---|
| `.claude-plugin/marketplace.json` | plugin 의 위치를 적은 marketplace catalog 이다. |
| `.claude/settings.json` | marketplace 를 등록하고 plugin 을 활성화하며, container 의 time zone 을 저자의 지역 시간대로 맞춘다. |

## 2. Marketplace Catalog

`.claude-plugin/marketplace.json` 은 plugin 하나를 `git-subdir` source 로 가리킨다. 원본 repo 전체가 아니라 `path` 에 적은 하위 folder 만 sparse clone 으로 받는다.

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

## 3. Settings

`.claude/settings.json` 의 `extraKnownMarketplaces` 는 catalog 를 어디서 읽을지 정하고, `enabledPlugins` 는 그 catalog 의 어떤 plugin 을 켤지 정한다. 두 값이 있어야 사용자가 `/plugin marketplace add` 를 직접 실행하지 않아도 새 세션에서 규칙이 실린다.

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

`enabledPlugins` 의 key 는 `<PLUGIN_NAME>@<MARKETPLACE_NAME>` 형식이며, 두 이름 모두 catalog 에 적은 값과 같아야 한다.

## 4. Setup For Another Repository

대상 repo 의 root 에서 위 두 파일을 만들면 끝난다. catalog 를 이 repo 에 그대로 두고 재사용하려면 `.claude/settings.json` 만 복사한다. plugin 이 담긴 repo 가 private 이면 각 machine 에 git credential helper 가 설정되어 있어야 fetch 가 성공한다.

```bash
# run from the target repository root
mkdir -p .claude
cp <SOURCE_REPO>/.claude/settings.json .claude/settings.json
```

commit 하고 push 한 뒤 새 세션을 열어 `/md_rules` 가 호출되면 plugin 이 실린 것이다.

## 5. Update

plugin 의 내용이 바뀌면 `/plugin marketplace update` 로 catalog 를 갱신한다. `ref` 가 branch 이므로 별도의 version 표기 없이 새 commit 이 곧 새 version 이 된다. 이 repo 에는 사본이 없으므로 따로 sync 할 파일이 없다.

## 6. Caution

같은 규칙을 `.claude/skills` 와 `.claude/hooks` 에 사본으로 두면 skill 이 두 벌 잡히고 대화 규칙도 두 번 주입된다. 이 repo 는 plugin 만 쓰므로 그 사본을 두지 않는다.

## Appendix A. Terminology

- **catalog**: plugin 의 이름과 위치를 나열한 `marketplace.json` 파일이다.
- **hook**: 정해진 시점에 Claude Code 가 실행하는 command 이다. UserPromptSubmit hook 의 출력은 prompt 마다 context 에 주입된다.
- **marketplace**: catalog 를 통해 plugin 을 배포하는 단위이다.
- **plugin**: skill, hook 등을 묶어 배포하는 단위이며, 세션 시작 시점에 설치되어 있어야 실린다.
- **skill**: `SKILL.md` 한 개로 정의하는 지시문 묶음이다. `/<name>` 으로 직접 호출하거나 Claude 가 필요할 때 스스로 로드한다.
- **sparse clone**: repo 의 일부 folder 만 내려받는 clone 방식이다.
