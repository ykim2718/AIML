> ⚠️ **This is an auto-synced copy. Do not edit here.**

# Automatic Rule Loading via Plugin Marketplace

rev. 93

## 1. Goal

이 문서는 Claude Code 전용으로 desktop과 web interface를 고려한다.

Automatic rule loading을 새 session과 새 machine에서 얻으려면 세 가지를 한 번 해 두면 된다.

1. plugin marketplace에 rule을 기록한다.
2. 그 marketplace를 remote git repository로 push 한다.
3. Desktop interface는 `~/.claude/settings.json` 에, Web interface는 repository에 함께 commit 되는 project settings file에 marketplace 등록과 plugin 활성화를 적어 bootstrap 한다.

그 뒤에는 두 가지가 저절로 이루어진다.

1. plugin이 새 session 시작 시 최신 상태로 update 된다. Desktop interface에서는 [4.4](#44-session-start-hook-desktop) 의 hook이 이 역할을 맡는다.
2. rule이 새 session과 새 prompt에 적용된다.

문서에서 자동이라고 할 때는 이 두 가지를 뜻한다.

## 2. Architecture

### 2.1 Copies and Their Roles

plugin marketplace의 사본이 여러 곳에 존재하지만 역할이 서로 다르다. 실행용 사본은 interface마다 별도로 존재한다.

```
             +------------------------------------------------------------+
             |  [1] GitHub                                                |
             |      Remote repository for plugin marketplace              |
             |      marketplace manifest and plugins                      |
             +------------------------------------------------------------+
               ^                           |                           |
       git push|            add and install|            add and install|
        manual |                 automatic |                 automatic |
               |                           v                           v
   +------------------------+  +------------------------+  +------------------------+
   |  [2] User machine      |  |  [3] Desktop interface |  |  [4] Web interface     |
   |      Working clone     |  |      Installed plugins |  |      Installed plugins |
   |                        |  |                        |  |                        |
   |  author edits          |  |  persists              |  |  discarded at session  |
   |  push to [1]           |  |                        |  |  end                   |
   +------------------------+  +------------------------+  +------------------------+
```

`[3]`이 내려받은 사본을 한 folder 아래에 나뉘어 저장한다. `[4]`의 저장 경로는 공식 문서에 없다.

```
~/.claude/plugins/
├── marketplaces/<name>/    : clone of <repository root>
└── cache/<...>/            : copy of each plugin folder -> plugin cache
```

`[2]`, `[3]`, `[4]`는 서로를 참조하지 않는다. 모두 `[1]`만 바라본다. Claude Code가 읽는 것은 `[3]` 또는 `[4]` 뿐이므로, working clone을 수정해도 push 전까지는 동작에 영향이 없다.

### 2.2 Marketplace and Plugin Layers in Remote Git Repository 🌳

marketplace는 catalog이고, plugin은 배포 단위이며, 실제 기능은 plugin 안의 component가 제공한다.

repository 최상위의 `.claude-plugin/marketplace.json` 은 자리가 정해져 있다. Claude Code가 repository를 내려받은 뒤 이 경로에서 catalog를 찾기 때문이며, 다른 곳에 두면 marketplace로 인식되지 않는다.

반면 `plugins/` 라는 folder 이름은 정해진 것이 아니라 관례이다. 각 plugin의 위치는 `marketplace.json` 의 `source` 값이 가리키므로 (`"source": "./plugins/<plugin-name>"`), folder 이름을 달리 하거나 다른 repository를 가리켜도 된다. catalog는 자리를 고정하고 내용물은 그 옆에 모아 두는 이 배치 덕분에, repository 하나가 plugin 여러 개를 담는 marketplace가 된다.

```
<repository root>/
├── .claude-plugin/
│   └── marketplace.json          : Marketplace manifest (catalog)
└── plugins/
    └── <plugin-name>/            : Plugin (distribution unit)
        ├── .claude-plugin/
        │   └── plugin.json       : Plugin manifest
        ├── skills/               : loaded on trigger
        ├── hooks/
        │   ├── hooks.json        : event binding
        │   │                       - UserPromptSubmit : every prompt
        │   │                       - SessionStart     : session start
        │   └── <rule>.md         : text injected into context
        ├── commands/             : slash commands
        └── agents/               : subagents
```

marketplace manifest는 repository 최상위의 `.claude-plugin/marketplace.json` 하나뿐이며, 어떤 plugin이 어디에 있는지만 나열한다.

plugin manifest는 각 plugin folder 안의 `.claude-plugin/plugin.json`으로, 그 plugin의 이름과 정보를 담는다. 실제 기능은 plugin folder 아래의 component folder가 담는다.

component 중 load 시점이 고정된 것은 hook뿐이다. 나머지는 조건이 맞을 때만 올라온다.

- **skill**: session에는 각 skill의 이름과 `description` 한 줄만 목록으로 떠 있다. 지금 하는 일이 그 설명과 들어맞는다고 Claude가 판단하면 그때 본문 file을 읽어 들인다. 그래서 `description`은 언제 쓰는 skill인지가 드러나게 적어야 하고, 그렇지 않으면 skill이 있어도 불려 나오지 않는다. 사용자가 `/<skill-name>` 을 입력하면 이 판단을 거치지 않고 곧바로 load 된다.
- **command**: 사용자가 prompt에 `/<command-name>` 을 입력할 때 그 file이 load 된다.
- **agent**: 사용자가 이름을 대어 지목하거나, Claude가 그 일을 subagent에 넘기기로 할 때 load 된다.
- **hook**: `hooks.json`에 UserPromptSubmit으로 묶은 file은 매 prompt마다 조건 없이 context에 들어간다.

반드시 지켜야 할 rule은 조건에 걸리지 않는 hook 경로에 둔다.

### 2.3 Bootstrap in Settings Files (Desktop/Web)

settings file에는 어느 marketplace를 받고 어느 plugin을 켤지가 적혀 있다. Claude Code는 그 내용대로 저장된 사본 안의 plugin을 session에 load 한다. 이 file은 두 자리에 있고, 이름은 같지만 적용 범위가 다르다.

```
User machine
├── ~/.claude/
│   └── settings.json           : machine settings - all projects on the machine
└── <project>/                  : git repository
    └── .claude/
        └── settings.json       : project settings - the project only
```

| Interface | Settings file | Coverage | Distribution |
|---|---|---|---|
| Desktop | `~/.claude/settings.json` | machine | git 밖의 개인 file이라 machine마다 직접 적는다 |
| Desktop | `<project>/.claude/settings.json` | project | commit 되어 repository와 함께 움직인다 |
| Web | `<project>/.claude/settings.json` | project | local file system이 없어 이 file만 읽는다 |

Desktop interface는 machine settings 덕분에 machine마다 한 번만 적으면 그 machine의 모든 project가 덮인다. machine settings와 project settings에 같은 내용이 있으면 병합되고, 겹치는 값은 project settings가 이긴다.

Web interface는 machine settings가 없다. 대신 remote repository의 project settings에 bootstrap이 push 되어 있으면, new session이 `add and install`을 다시 하므로 매 session 최신 marketplace를 받는다. Desktop interface처럼 [4.4](#44-session-start-hook-desktop) 의 hook을 따로 둘 필요도 없다. 다만 remote repository의 project settings가 비어 있으면 rule은 올라오지 않는다.

## 3. Setup

이 절은 [2](#2-architecture) 의 구조를 실제로 만드는 순서를 다룬다. 먼저 remote git repository에 marketplace를 만들고 (3.1), 그다음 settings file에 bootstrap을 적는다 (3.2). 적는 내용은 하나이며, 그 내용을 어느 settings file에 두는가에 따라 적용 범위가 달라진다.

### 3.1 Remote Git Repository

rule을 담는 plugin marketplace는 일반 git repository이며, 최소 구성은 manifest file 두 개이다.

```
<repository root>/
├── .claude-plugin/
│   └── marketplace.json          : marketplace manifest
└── plugins/
    └── <plugin-name>/
        └── .claude-plugin/
            └── plugin.json       : plugin manifest
```

이 구조만 갖추어 GitHub에 push 하면 등록 가능한 marketplace가 된다.

두 manifest의 실제 예시는 다음과 같다.

`.claude-plugin/marketplace.json` — 어떤 plugin이 어디에 있는지 나열한다:

```json
{
  "name": "claude-configuration",
  "owner": { "name": "yRocket", "email": "ykim2718@gmail.com" },
  "plugins": [
    {
      "name": "yrocket-rules",
      "source": "./plugins/yrocket-rules",
      "description": "코드/문서 작성 공용 규칙 skill과 hook."
    }
  ]
}
```

`plugins/yrocket-rules/.claude-plugin/plugin.json` — plugin 자신의 이름과 정보를 담는다:

```json
{
  "name": "yrocket-rules",
  "description": "코드/문서 작성 공용 규칙: coding_rules·md_doc_rules skill, 대화 규칙과 필수 skill 로딩을 주입하는 UserPromptSubmit hook."
}
```

⛔ **`version` field는 넣지 않는다.** `version`을 적으면 그 값이 바뀔 때까지 plugin이 갱신되지 않는다 (pin). 값을 그대로 두고 내용만 push 하면 Claude Code는 같은 version으로 판정하여 cache 사본을 유지한다. `version`을 생략하면 commit SHA가 version이 되어, push 할 때마다 새 version으로 인식되고 자동으로 갱신된다.

### 3.2 Settings Files (Desktop/Web)

bootstrap을 어느 file에 적을지는 [2.3](#23-bootstrap-in-settings-files-desktopweb) 의 표대로 고른다. Desktop interface만 쓰면 `~/.claude/settings.json` 한 곳으로 그 machine의 모든 project가 덮이고, Web interface까지 덮으려면 commit 되는 `<project>/.claude/settings.json` 에 적어야 한다.

어느 file이든 적는 내용은 marketplace 위치와 plugin 활성화로 같다.

```json
{
  "extraKnownMarketplaces": {
    "claude-configuration": {
      "source": {
        "source": "github",
        "repo": "ykim2718/Claude-Configuration"
      },
      "autoUpdate": true
    }
  },
  "enabledPlugins": {
    "yrocket-rules@claude-configuration": true
  }
}
```

**`"autoUpdate": true` 는 marketplace를 GitHub 최신 기준으로 갱신하라는 지시이다.** 나머지 field는 무엇을 받을지 가리킬 뿐이고, 갱신 대상을 정하는 것은 이 값이다.

⛔ **Desktop interface에서는 이 한 줄만으로 갱신되지 않는다.** Desktop app이 session 프로세스에 `DISABLE_AUTOUPDATER=1` 을 심으므로 plugin 자동 갱신이 통째로 skip 된다. 자세한 내용과 대응은 [4.1](#41-session-level-update) 과 [4.4](#44-session-start-hook-desktop) 를 본다.

settings file에 적지 않고 session에서 한 번만 설치할 수도 있다. prompt에 다음을 입력하면 각각 marketplace 등록과 plugin 설치가 일어난다. 단, `"autoUpdate": true` 와 `"yrocket-rules@claude-configuration": true` 는 수동으로 기입해야 한다.

```
# prompt
/plugin marketplace add ykim2718/Claude-Configuration
/plugin install yrocket-rules@claude-configuration
```

settings file 수정도 손으로 할 필요 없이 prompt에서 지시하면 된다. Claude가 위의 JSON과 같은 내용을 만들어 넣는다.

```
# prompt
project의 .claude/settings.json에 extraKnownMarketplaces로
ykim2718/Claude-Configuration (github source, autoUpdate: true)을
claude-configuration 이름으로 등록하고,
enabledPlugins에 yrocket-rules@claude-configuration: true 를 추가해줘.
```

machine 전체에 적용하려면 "project의 .claude/settings.json" 대신 "`~/.claude/settings.json`"이라고 지시하면 된다.

settings file에 손으로 적어야 하는 것은 `extraKnownMarketplaces` 와 `enabledPlugins` 두 항목뿐이다. 이 둘은 marketplace를 처음 가리키는 bootstrap이라 marketplace 자신이 배포할 수 없다. 그 뒤의 rule과 갱신 hook은 [4.4](#44-session-start-hook-desktop) 처럼 repository가 배포하므로 push만으로 따라온다.

#### 3.2.1 Field Reference

| Field | Role |
|---|---|
| `extraKnownMarketplaces` | marketplace의 이름과 source를 등록한다 |
| `source.source` | source type을 지정하며 `github`, `git`, `url`, `npm`, `file`, `directory` 를 지원한다 |
| `autoUpdate` | marketplace와 plugin을 session 시작 시 갱신 대상으로 삼는다. Desktop interface에서는 실행되지 않는다 |
| `enabledPlugins` | `plugin-name@marketplace-name` 형식의 key를 `true`로 두어 활성화한다 |

field의 이름과 의미는 어느 settings file에 두든 동일하다. 두 file에 함께 적었을 때의 우선순위는 [2.3](#23-bootstrap-in-settings-files-desktopweb) 대로 `machine settings < project settings` 이며, 그 위에 실행할 때 지정하는 option과 OS 단위로 배포하는 managed settings가 있다. managed settings는 Desktop interface에만 도달한다.

## 4. Automatic Update

### 4.1 Session-Level Update

`autoUpdate`를 `true`로 두면 Claude Code가 session 시작 시 `[1]`을 기준으로 사본을 갱신한다. 다만 Desktop interface에서는 이 갱신이 실행되지 않는다.

| Aspect | Desktop interface | Web interface |
|---|---|---|
| 갱신 시점 | 실행되지 않는다 | session 시작 |
| `autoUpdate: true`의 역할 | 무력하다 | 갱신할 이전 사본이 없어 무의미하다 |
| 결과 | 설치 당시 commit에 고정된다 | 최신 commit |

Desktop app은 자신의 update를 스스로 관리하므로 session 프로세스에 `DISABLE_AUTOUPDATER=1` 을 심는데, plugin 갱신이 같은 auto-updater 경로에 얹혀 있어 함께 꺼진다. debug log에는 다음 한 줄이 남는다.

```text
[DEBUG] Plugin autoupdate: skipped (auto-updater disabled)
```

이 변수는 settings file의 `env` block으로 덮이지 않는다. app이 값을 나중에 적용하므로 새 session에서도 `1` 이 유지된다. 따라서 Desktop interface에서 갱신을 자동화하려면 [4.4](#44-session-start-hook-desktop) 의 hook을 쓴다.

### 4.2 Prompt-Level Injection

marketplace 갱신이 session 단위인 것과 달리, UserPromptSubmit hook은 prompt 단위로 동작한다. 이 hook이 repository를 다시 읽는 것은 아니며, 이미 내려받아 둔 사본의 내용을 매 prompt마다 context에 넣는다.

| Aspect | Desktop interface | Web interface |
|---|---|---|
| 정의 위치 | plugin의 `hooks/` | 동일 |
| 실행 시점 | 매 prompt | 매 prompt |
| 읽는 대상 | `[3]` 설치된 plugin | `[4]` 설치된 plugin |
| `[1]` 재조회 | 없다 | 없다 |

따라서 rule을 push 한 뒤 열려 있는 session에서 prompt를 반복해도 새 내용은 들어오지 않는다. 두 interface 모두 새 session을 열어야 반영된다.

### 4.3 Manual Force Update (Desktop)

⚠️ Desktop interface에서는 autoUpdate가 실행되지 않으므로, 갱신은 사람이 시키거나 [4.4](#44-session-start-hook-desktop) 의 hook이 대신해야 한다. push 내용을 반영하려면 terminal에서 Claude CLI로 다음 두 명령을 차례로 실행한다. Desktop interface의 prompt에서는 `/plugin` 명령이 동작하지 않을 수 있으므로 terminal CLI를 사용하며, CLI 설치는 [Appendix B](#appendix-b-claude-cli) 를 본다.

```bash
# claude CLI
claude plugin marketplace update <MARKETPLACE_NAME>
claude plugin update <PLUGIN_NAME>@<MARKETPLACE_NAME>
```

갱신은 두 단계이며 앞 단계만으로는 session에 반영되지 않는다.

| Step | Command | Effect |
|---|---|---|
| 1 | `claude plugin marketplace update` | marketplace clone을 최신 commit으로 옮긴다 |
| 2 | `claude plugin update` | 설치본을 새 commit의 cache 사본으로 다시 고정한다 |

`installed_plugins.json` 의 `gitCommitSha` 가 marketplace clone의 HEAD와 같아지면 갱신이 끝난 것이다. 갱신된 plugin은 현재 열려 있는 session에는 적용되지 않고, 다음 session부터 적용된다.

### 4.4 Session Start Hook (Desktop)

SessionStart hook에 [4.3](#43-manual-force-update-desktop) 의 두 명령을 걸면 사람이 개입하지 않아도 session을 열 때마다 갱신이 실행된다. Desktop interface에서 autoUpdate를 대신하는 자리이다.

이 hook은 plugin의 `hooks/hooks.json` 에 둔다. 그러면 hook 자신이 marketplace를 통해 배포되므로, 다른 컴퓨터는 plugin을 설치하는 것만으로 같은 갱신 동작을 얻는다. UserPromptSubmit과 같은 file에 나란히 놓이며, 전문은 다음과 같다.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cat \"${CLAUDE_PLUGIN_ROOT}/hooks/conversation_rules.md\" \"${CLAUDE_PLUGIN_ROOT}/hooks/skill_loading_rules.md\""
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "{ command -v claude >/dev/null 2>&1 || { echo \"claude CLI not found; skipping plugin self-update\"; exit 0; }; state=\"$HOME/.claude/plugins/installed_plugins.json\"; [ -f \"$state\" ] || { echo \"no installed plugins\"; exit 0; }; date \"+=== session start %Y-%m-%d %H:%M:%S\"; claude plugin marketplace update; for p in $(grep -oE '\"[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+\"' \"$state\" | tr -d '\"' | sort -u); do claude plugin update \"$p\"; done; } >> \"$HOME/.claude/plugin-autoupdate.log\" 2>&1",
            "timeout": 180
          }
        ]
      }
    ]
  }
}
```

작성할 때 지켜야 할 조건은 세 가지이며, 지키지 않으면 hook이 조용히 실패한다.

| Condition | Reason |
|---|---|
| 명령을 bash 문법으로 쓴다 | hook은 OS와 무관하게 bash로 실행된다. PowerShell이나 cmd 문법의 pipe와 redirect는 bash가 다르게 해석하여 아무 일도 일어나지 않는다 |
| 경로를 `$HOME` 으로 쓴다 | `%USERPROFILE%` 은 bash가 확장하지 않는다 |
| 이름을 하드코딩하지 않는다 | `installed_plugins.json` 에서 읽으면 marketplace와 plugin이 늘어도 그대로 동작한다 |

세 위치 모두에서 hook이 실행되지만, 배포 여부가 다르므로 plugin의 `hooks.json` 을 기본으로 삼는다.

| Location | Distributed by push | Applies to a new machine |
|---|---|---|
| plugin의 `hooks/hooks.json` | 예 | plugin 설치만으로 적용된다 |
| project의 `.claude/settings.json` | 예 | 그 project를 열면 적용된다 |
| `~/.claude/settings.json` | 아니오 | 손으로 적어야 한다 |

같은 hook을 두 곳에 두면 session마다 갱신이 두 번 실행되므로, plugin으로 옮긴 뒤에는 settings file 쪽을 지운다. matcher를 `startup` 으로 두면 새 session에서만 실행되고 resume에서는 실행되지 않는다.

동작 여부는 log 파일로 확인한다.

```bash
grep -c '=== session start' ~/.claude/plugin-autoupdate.log
```

session을 열 때마다 이 값이 늘면 정상이다. hook이 실행한 갱신 역시 그 session이 아니라 다음 session부터 적용되므로, 한 session의 지연은 남는다.

## 5. Verification

### 5.1 Session Command

Claude Code의 Desktop과 Web interface에서 prompt에 다음을 입력하면 설치된 marketplace와 plugin 목록을 확인할 수 있다.

```
# prompt
Show known_marketplaces.json
Show installed_plugins.json
```

두 file의 상세는 [Appendix E](#appendix-e-plugin-state-files) 에 있다.

### 5.2 Claude CLI

Desktop interface에서는 `claude` CLI로도 확인할 수 있다. shell 종류와 무관하므로 PowerShell, bash 등 아무 terminal에서나 실행한다. 단, CLI는 별도 설치가 필요하다. 설치 방법은 [Appendix B](#appendix-b-claude-cli) 를 본다.

```
# claude CLI
claude plugin marketplace list
claude plugin list
claude plugin details yrocket-rules@claude-configuration
```

`details` 결과에 skill과 hook이 나타나면 정상이다.

## 6. Extension

repository에는 Claude Code가 읽는 경로가 정해져 있다. 그 밖의 file과 folder는 무시되므로 자유롭게 추가할 수 있다. 이 절의 확장이 놓이는 자리는 다음과 같다.

```
<repository root>/
├── .claude-plugin/
│   └── marketplace.json          : register when adding a plugin
├── plugins/
│   └── <plugin-name>/
│       ├── .claude-plugin/
│       │   └── plugin.json
│       ├── hooks/
│       │   ├── hooks.json
│       │   └── <rule>.md
│       ├── commands/
│       │   └── <command>.md
│       ├── agents/
│       │   └── <agent>.md
│       └── skills/
│           └── <skill-name>/
│               ├── SKILL.md
│               └── references/
│                   └── <topic>.md
└── docs/                         : non-loaded document
```

### 6.1 Hook

`hooks/` 에 rule file을 추가하고 `hooks.json`에 event를 연결한다. UserPromptSubmit에 묶으면 매 prompt마다 조건 없이 context에 들어간다. 같은 file에 SessionStart를 함께 묶을 수 있으며, plugin이 자신을 갱신하는 hook이 그 자리에 놓인다.

### 6.2 Command

`commands/` 에 md file을 추가하면 file 이름이 slash command 이름이 된다. 사용자가 호출할 때만 load 된다.

### 6.3 Agent

`agents/` 에 md file을 추가한다. subagent의 정의이며, 호출될 때만 load 된다.

### 6.4 Plugin

`plugins/` 아래에 새 plugin folder를 만들고 `.claude-plugin/plugin.json`을 둔다. `marketplace.json` 등록이 필요한 유일한 확장이다.

### 6.5 Skill and Reference File

`skills/<skill-name>/SKILL.md` 를 추가한다. folder 이름이 skill 이름이 되고, 상세 내용은 `references/` 로 분리한다. 작성 방법은 [Appendix D](#appendix-d-skill) 를 본다.

### 6.6 Non-Loaded Document

설계 memo나 참고 자료처럼 Claude Code가 읽을 필요가 없는 문서는 plugin 구조 밖에 둔다. `docs/` 같은 folder는 무시되므로 동작에 영향을 주지 않는다.

repository 전체가 실행용 사본으로 복사되므로 용량이 큰 file은 피한다. 필요하면 3.2절의 `.claude/settings.json`에 있는 `source` object에 `"sparsePaths": [<path>, ...]` 를 지정하여 일부 folder만 받도록 제한할 수 있다.

## 7. Constraints

- plugin에 담을 수 있는 component는 skill, hook, command, agent뿐이다. CLAUDE.md는 plugin에 담을 수 없으므로, 반드시 지켜야 할 지시는 매 prompt 주입되는 UserPromptSubmit hook에 둔다.
- 저장된 CLAUDE.md를 치환하는 것은, hook이 임의의 command를 실행할 수 있으므로, SessionStart hook에 "cache의 CLAUDE.md를 project로 복사"를 시켜서 기술적으로 가능하다. 하지만 CLAUDE.md는 session 시작 시 읽히는데, hook도 session 시작 시 돌므로 복사 결과가 이번 session에 잡힌다는 보장이 없다.
- private repository는 GitHub 인증이 된 환경에서만 설치된다. 설치가 실패하면 repository 공개 범위를 확인한다.
- 조직 단위로 배포하는 server-managed settings는 두 interface에 모두 적용되며, Team이나 Enterprise plan에서 owner 또는 admin이 claude.ai의 admin 화면에서 설정한다. `enabledPlugins`로 특정 plugin을 강제할 수는 있으나 `extraKnownMarketplaces`를 이 경로로 배포하는 방법은 문서에 없으므로, marketplace 등록은 여전히 project settings가 맡는다.

## Appendix A. Terminology

- **`autoUpdate`**: marketplace 등록 항목의 field이다. session 시작 시 marketplace와 plugin을 remote 기준으로 갱신하지만, Desktop interface에서는 `DISABLE_AUTOUPDATER=1` 때문에 실행되지 않는다.

- **`DISABLE_AUTOUPDATER`**: Desktop app이 session 프로세스에 심는 환경변수이다. 값이 `1` 이면 plugin 자동 갱신이 skip 되며, settings file의 `env` block으로 덮이지 않는다.

- **claude.ai**: Claude 계정으로 접속하는 web service이다. Web interface는 이 service 안에서 열리며, 개인 skill을 켜고 끄는 화면과 조직의 server-managed settings를 다루는 admin 화면도 여기에 있다. 두 interface가 코드를 다루는 곳이라면, claude.ai는 계정과 조직을 다루는 곳이다.

- **Claude Code in the Desktop**: local machine에서 실행하는 interface이다. `~/.claude/` 아래의 machine settings와 plugin cache를 사용한다. Desktop interface로 줄여 쓴다.

- **Claude Code on the Web**: cloud에서 repository를 clone 하여 실행하는 interface이다. local file system이 없으므로 repository에 commit 된 설정만 적용된다. Web interface로 줄여 쓴다.

- **Context**: model이 응답을 만들 때 참조하는 입력 전체이다. 대화 내용과 함께 load 된 skill, hook이 주입한 rule이 여기에 들어간다. 크기에 한계가 있으므로 항상 load 되는 file은 짧게 유지한다.

- **Global settings**: `~/.claude/settings.json` 이다. 사용자가 자기 machine에 두는 file이므로 그 machine의 모든 project에 적용되고, Web interface에는 존재하지 않는다.

- **Hook**: 지정한 event에 개입하는 실행 지점이다. session 시작 시 동작하는 SessionStart hook과 매 prompt마다 동작하는 UserPromptSubmit hook을 사용한다.

- **Marketplace**: plugin의 catalog이다. `.claude-plugin/marketplace.json` 하나로 정의하며, 어떤 plugin이 어디에 있는지 나열한다.

- **Plugin**: 배포 단위이다. `.claude-plugin/plugin.json`에 자신의 이름과 정보를 두고, 하위 folder에 skill, hook, command, agent를 담는다.

- **Plugin cache**: Desktop interface가 내려받은 plugin을 실행하는 위치인 `~/.claude/plugins/cache/` 이다. 원본을 그 자리에서 쓰지 않고 이곳으로 복사해 실행한다. Claude Code가 관리하므로 직접 수정하지 않는다.

- **Plugin marketplace**: marketplace manifest와 plugin들을 함께 담은 git repository 전체이다. rule은 이 안에서 관리되고, 각 interface는 이것을 내려받아 사용한다.

- **Project in Claude Code**: 작업 중인 directory 그 자체이며, 보통 git repository이다. 별도로 등록하는 절차가 없고, Claude Code를 연 위치가 곧 project가 된다. `.claude/` 와 CLAUDE.md도 그 directory를 기준으로 찾는다. Web interface에서는 clone 된 repository가 project가 된다.

- **Project settings**: project의 `.claude/settings.json` 이다. commit 되므로 두 interface와 다른 사용자에게 모두 적용된다.

- **Prompt**: session 안에서 사용자가 보내는 한 번의 입력이다. session보다 작은 단위이며, UserPromptSubmit hook은 이 단위로 동작한다.

- **Server-managed settings**: 조직 단위로 배포되는 settings이다. Team이나 Enterprise plan의 조직에서 owner 또는 admin이 claude.ai의 admin 화면에서 설정하며, session 시작 시 서버에서 내려와 두 interface에 모두 적용된다. OS 단위로 배포하는 managed settings file과 달리 Web interface까지 도달한다.

- **Session**: 하나의 Claude Code 실행 단위이다. 시작 시점에 settings, plugin, CLAUDE.md, hook 정의를 읽어 들이고, 그 구성으로 대화가 끝날 때까지 동작한다. rule 변경이 반영되는 경계가 곧 session이다.

- **Skill**: 필요한 시점에 load 되는 rule 묶음이다. folder 이름이 skill 이름이 되고, `SKILL.md`의 `description`이 load 시점을 결정한다.

- **Working clone**: rule을 수정하기 위해 받아 둔 plugin marketplace의 clone이다. 실행에는 쓰이지 않으며, push를 통해서만 동작에 반영된다.

- **Working directory**: Claude Code를 실행한 folder이다. 이 folder가 곧 project가 되며, `.claude/` 와 CLAUDE.md도 이 위치를 기준으로 찾는다.

## Appendix B. Claude CLI

### B.1 CLI Installation

Desktop app을 설치해도 `claude` CLI는 PATH에 들어오지 않는다. terminal에서 `claude` 명령을 쓰려면 별도로 설치한다.

| Environment | Command |
|---|---|
| Windows (PowerShell) | `# Windows shell`<br>`irm https://claude.ai/install.ps1 \| iex` |
| macOS / Linux / WSL | `# Bash`<br>`curl -fsSL https://claude.ai/install.sh \| bash` |

WSL은 Windows와 별개의 환경이므로 양쪽에서 쓰려면 각각 설치한다. 설치 후 terminal을 새로 열고 `claude --version` 으로 설치를 확인한다.

### B.2 CLI Commands

terminal에서 실행하는 plugin 관련 명령들이다.

- **`claude plugin marketplace add <OWNER>/<REPO>`**: plugin marketplace를 등록한다.
- **`claude plugin install <PLUGIN_NAME>@<MARKETPLACE_NAME>`**: plugin을 설치한다.
- **`claude plugin marketplace update <MARKETPLACE_NAME>`**: marketplace와 plugin을 수동으로 갱신한다.
- **`claude plugin update <PLUGIN_NAME>@<MARKETPLACE_NAME>`**: 설치된 plugin을 수동으로 갱신한다.

이 repository에 적용하는 예시는 다음과 같다.

```bash
# claude CLI
claude plugin marketplace update claude-configuration
claude plugin update yrocket-rules@claude-configuration
```

## Appendix C. Prompt Command

session의 prompt에 입력하는 명령들이다.

- **`/<skill-name>`**: `description`의 판단과 무관하게 그 skill을 바로 load 한다.

`/plugin` 명령이 prompt에서 동작하지 않으면, 아래처럼 CLI 실행을 지시하는 prompt를 입력한다.

```
# prompt
claude plugin marketplace update claude-configuration 과
claude plugin update yrocket-rules@claude-configuration 을 실행해줘.
```

## Appendix D. Skill

### D.1 Skill

새 rule 묶음은 skill folder를 추가하는 것으로 끝난다. plugin manifest는 수정하지 않아도 된다.

file 이름은 반드시 `SKILL.md`이며, folder 이름이 skill 이름이 된다. frontmatter의 `description`이 언제 이 skill을 load 할지 판단하는 근거이므로, 적용 시점을 분명히 적는다.

```markdown
---
name: <skill-name>
description: 적용 대상과 load 시점을 명시한다.
---
```

사용자가 prompt에 `/<skill-name>` 을 입력하면 `description`의 판단과 무관하게 그 skill이 바로 load 된다.

### D.2 Reference File

skill의 상세 내용은 별도 file로 분리하고 SKILL.md에서 가리킨다. SKILL.md는 항상 load 되지만 reference file은 필요할 때만 읽히므로, 기본 context 소비를 줄이면서 깊은 내용에 도달할 수 있다.

```
skills/<skill-name>/
├── SKILL.md              (always loaded, keep short)
└── references/
    └── <topic>.md        (loaded on demand)
```

## Appendix E. Plugin State Files

`~/.claude/plugins/` 아래의 두 file은 Claude Code가 plugin 기능을 실제로 사용할 때 자동으로 생성하고 관리하는 내부 상태 file이다.

- **`known_marketplaces.json`**: marketplace를 처음 추가할 때 (`/plugin marketplace add` 를 실행하거나, settings file의 `extraKnownMarketplaces` 항목을 읽어 clone 할 때) 생성된다. marketplace를 하나도 등록한 적 없는 새 설치 환경에는 없다.
- **`installed_plugins.json`**: plugin을 처음 설치할 때 생성된다. 마찬가지로 설치한 plugin이 없으면 없다.

역할 구분은 다음과 같다.

| File | Role |
|---|---|
| settings file의 `extraKnownMarketplaces`, `enabledPlugins` | 사용자가 적는 설정 (무엇을 쓰겠다) |
| `~/.claude/plugins/` 의 `known_marketplaces.json`, `installed_plugins.json` | Claude Code가 기록하는 상태 (실제로 무엇이 언제 어디에 설치됐다) |

그래서 settings file은 직접 편집해도 되지만, `~/.claude/plugins/` 아래의 상태 file은 Claude Code가 관리하므로 직접 손대지 않는 게 좋다. 지우더라도 다음에 marketplace와 plugin을 다시 등록하면 재생성된다.

실제 예시는 다음과 같다.

`known_marketplaces.json` — 등록된 marketplace와 그 clone 위치, 마지막 갱신 시각을 기록한다:

```json
{
  "claude-configuration": {
    "source": {
      "source": "github",
      "repo": "ykim2718/Claude-Configuration"
    },
    "installLocation": "C:\\Users\\Asus\\.claude\\plugins\\marketplaces\\claude-configuration",
    "lastUpdated": "2026-07-21T01:21:01.491Z",
    "autoUpdate": true
  }
}
```

`installed_plugins.json` — 설치된 plugin의 version, 설치 시각, 설치 당시의 commit을 기록한다. plugin.json에 `version` field가 없으면 commit SHA가 version으로 기록된다:

```json
{
  "version": 2,
  "plugins": {
    "yrocket-rules@claude-configuration": [
      {
        "scope": "user",
        "installPath": "C:\\Users\\Asus\\.claude\\plugins\\cache\\claude-configuration\\yrocket-rules\\ad5437a04a0c79d561ce26748a2fa01f5b37c617",
        "version": "ad5437a04a0c79d561ce26748a2fa01f5b37c617",
        "installedAt": "2026-07-21T01:23:23.579Z",
        "lastUpdated": "2026-07-21T01:23:23.579Z",
        "gitCommitSha": "ad5437a04a0c79d561ce26748a2fa01f5b37c617"
      }
    ]
  }
}
```

## Appendix F. Obsidian

Obsidian은 local folder에 놓인 md file을 그대로 읽고 쓰는 편집 도구이다. 자체 file 형식이나 database를 두지 않고, folder 하나를 vault로 지정해 그 안의 md file을 note로 다룬다. file 사이의 link를 따라가거나 전체를 한 번에 검색하는 기능이 편집기와 다른 점이다.

rule은 결국 md file이므로 Obsidian으로 편집할 수 있다. vault에 별도의 형식이 없으므로 working clone을 vault로 삼으면 그대로 열린다.

### F.1 Vault Placement

배치는 세 가지가 있다.

| Placement | Description | Note |
|---|---|---|
| working clone 자체를 vault로 연다 | rule만 담긴 독립 vault가 된다 | 가장 단순하며 다른 기록과 섞이지 않는다 |
| 기존 vault 안에 working clone을 둔다 | vault 하위 folder가 git repository가 된다 | 기존 note와 함께 검색되지만, vault 전체를 sync 하는 도구와 git이 같은 file을 건드린다 |
| vault 밖에 두고 symlink를 건다 | 실체는 vault 밖에 있고 vault에는 link만 둔다 | 배치는 자유롭지만 Obsidian이 link 대상을 vault 경계 밖으로 인식하는 경우가 있다 |

기록용 vault를 이미 쓰고 있다면 첫 번째를 권한다. rule repository는 push 시점이 곧 배포 시점이라, 일반 note와 commit 주기를 섞지 않는 편이 안전하다.

### F.2 Link Style

Obsidian의 기본 link 형식인 `[[wikilink]]` 는 Claude Code가 해석하지 않는다. skill이 reference file을 가리키는 link는 상대 경로 markdown link로 적는다.

```markdown
좋음: 자세한 내용은 [naming](references/naming.md) 을 본다.
나쁨: 자세한 내용은 [[naming]] 을 본다.
```

Obsidian 설정에서 wikilink를 끄고 link를 상대 경로로 두면, 새로 만드는 link도 같은 형식이 된다. vault 안에서만 쓰는 note끼리는 wikilink를 써도 무방하지만, plugin folder 아래 file에는 쓰지 않는다.

### F.3 File Hygiene

working clone 자체를 vault로 열었을 때의 구조는 다음과 같다. Obsidian이 더하는 것은 최상위의 `.obsidian/` 하나뿐이다.

```
<working clone> = vault root
├── .obsidian/                : Obsidian settings, not read by Claude Code
│   ├── app.json              : editor settings, link format set here
│   ├── workspace.json        : window layout, changes often
│   └── plugins/              : Obsidian community plugin
├── .claude-plugin/
│   └── marketplace.json
├── plugins/                  : Claude Code plugins, keep attachments out
│   └── <plugin-name>/
│       ├── .claude-plugin/
│       │   └── plugin.json
│       └── skills/
│           └── <skill-name>/
│               ├── SKILL.md
│               └── references/
└── docs/
    ├── <document>.md
    └── assets/               : set attachment location here
```

`plugins` 라는 이름이 두 곳에 나오지만 서로 관계가 없다. `.obsidian/plugins/` 는 Obsidian이 쓰는 것이고, 최상위의 `plugins/` 는 Claude Code가 쓰는 것이다.

`.obsidian/` 은 혼자 쓰는 vault라면 `.gitignore` 에 넣고, 설정을 함께 쓰고 싶다면 commit 한다.

첨부 file이 생길 자리를 미리 지정한다. 지정하지 않으면 note를 만든 위치, 즉 plugin folder 안에 image가 쌓인다.

frontmatter를 자동으로 정리하는 부류의 community plugin은 SKILL.md에 적용하지 않는다. `name` 과 `description` 은 skill이 언제 load 될지 정하는 값이라, key 순서나 표기가 바뀌면 의도와 다르게 동작할 수 있다.

### F.4 Sync

vault 전체를 장치 사이에서 자동으로 복제하는 기능이 있다. Obsidian이 제공하는 sync service, vault folder를 cloud drive 안에 두는 방식, folder를 실시간으로 맞추는 file 동기화 도구가 모두 여기에 해당한다. 이 기능을 모두 끄고 Obsidian은 편집기로만 사용하며, rule repository는 git의 commit과 push로만 관리하여 원본이 remote repository에 있도록 한다.
