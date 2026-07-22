> ⚠️ **This is an auto-synced copy.** Do not edit here.

# Automatic Rule Loading via Plugin Marketplace

rev. 60

## 1. Goal

Automatic rule loading은 rule을 한 repository에서 관리하고, 각 project는 settings file에 그 repository를 한 번 적어 두는 것으로 끝내는 방식이다. 그 뒤에는 두 가지가 저절로 이루어진다.

1. plugin이 새 session 시작 시 최신 상태로 update 된다.
2. rule이 새 session과 새 prompt에 적용된다.

문서에서 자동이라고 할 때는 이 두 가지를 뜻한다.

이 문서는 Claude Code in the Desktop (Desktop interface) 과 Claude Code on the Web (Web interface) 에서 동일한 plugin (skills, hooks, commands, agents) 을 재사용하기 위해, plugin marketplace를 구성하고 적용하는 방법을 정리한다.
## 2. Problem

기존 방식에는 두 가지 문제가 있다.

1. **CLAUDE.md에 GitHub URL을 적어 두는 방식은 동작하지 않는다.** CLAUDE.md는 local file import (`@path`) 만 지원하며, 원격 URL은 단순 text로 남는다. 새 session이 시작해도 해당 repository를 clone 하거나 내용을 가져오지 않는다.

2. **project마다 skill과 hook을 복사해 두는 방식은 유지가 어렵다.** rule을 고칠 때마다 모든 project를 찾아다녀야 하고, 사본 간에 내용이 어긋난다.

plugin marketplace는 이 두 문제를 함께 해결한다. rule을 한 곳에서 관리하고, 각 project는 참조만 한다.

## 3. Architecture

### 3.1 Copies and Coverage

marketplace repository의 사본이 여러 곳에 존재하지만 역할이 서로 다르다. 실행용 사본은 interface마다 별도로 존재한다.

```
             +------------------------------------------------------------+
             |  [1] GitHub                                                |
             |      Remote repository                                     |
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

User machine의 working clone `[2]` 에서 rule을 수정하고 commit/push 하면, 이후의 새 session부터 자동으로 적용된다.

`[3]`이 내려받은 사본을 한 folder 아래에 나뉘어 저장한다. `[4]`의 저장 경로는 공식 문서에 없다.

```
~/.claude/plugins/
├── marketplaces/<name>/    : clone of <repository root>
└── cache/<...>/            : copy of each plugin folder -> plugin cache
```

저장된 사본을 Claude Code가 다음 Edit location에 적힌 내용에 따라 적용한다.

| Site | Edit location | Coverage |
|---|---|---|
| `[1]` GitHub | marketplace repository | - |
| `[2]` User machine | workspace folder | - |
| `[3]` Desktop interface | `~/.claude/settings.json`<br>`<project>/.claude/settings.json` | machine<br>project |
| `[4]` Web interface | `<project>/.claude/settings.json` | project |

Edit location은 수정이 일어나는 자리이고, Coverage는 그 수정이 적용되는 범위이다. `[4]`의 설치는 session이 끝나면 사라지므로, 다음 session이 `add and install`을 다시 한다.

같은 내용을 여러 settings file에 함께 적으면 병합되고, 겹치는 값은 project settings가 이긴다.

account 단위 settings file은 어느 interface에도 없다. Desktop interface는 machine 단위 global settings가 그 자리를 대신해, machine마다 한 번 적으면 그 machine의 모든 project가 덮인다. Web interface에는 machine이 없어 global settings도 없으며, project settings이 비어 있는 project를 열면 rule은 올라오지 않는다. claude.ai 설정 화면에서 켠 개인 skill이 Web interface의 session에 들어오기는 하지만, 이는 skill일 뿐 plugin과 marketplace가 아니다.

조직 단위로는 server-managed settings가 두 interface에 모두 적용된다. Team이나 Enterprise plan에서 owner 또는 admin이 claude.ai의 admin 화면에서 설정한다. 다만 `enabledPlugins`로 특정 plugin을 강제하는 것은 가능하나 `extraKnownMarketplaces`를 이 경로로 배포하는 방법은 문서에 없으므로, marketplace 등록은 여전히 project settings가 맡는다.

`[2]`, `[3]`, `[4]`는 서로를 참조하지 않는다. 모두 `[1]`만 바라본다. Claude Code가 읽는 것은 `[3]` 또는 `[4]` 뿐이므로, working clone을 수정해도 push 전까지는 동작에 영향이 없다.

### 3.2 Marketplace and Plugin Layers

marketplace는 catalog이고, plugin은 배포 단위이며, 실제 기능은 plugin 안의 component가 제공한다.

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

marketplace manifest는 repository 최상위의 `.claude-plugin/marketplace.json` 하나뿐이며, 어떤 plugin이 어디에 있는지만 나열한다. plugin manifest는 각 plugin folder 안의 `.claude-plugin/plugin.json`으로, 그 plugin의 이름과 정보를 담는다. 실제 기능은 plugin folder 아래의 component folder가 담는다.

component 중 load 시점이 고정된 것은 hook뿐이다. skill은 `description`이 맞을 때, command와 agent는 호출될 때 load 되지만, `hooks.json`에 UserPromptSubmit으로 묶은 file은 매 prompt마다 조건 없이 context에 들어간다. 반드시 지켜야 할 rule을 이 경로에 둔다.

### 3.3 Repository Creation

marketplace repository는 정해진 자리에 manifest file 두 개를 둔 일반 git repository이다. 빈 folder에서 손으로 만들 필요 없이, 이미 동작하는 repository를 Claude Code에 시켜 복사하는 것이 가장 쉽다.

1. GitHub에서 빈 repository를 만든다.
2. Claude Code session을 열고 지시한다:  
   "`ykim2718/Claude-Configuration` 을 clone 해서 marketplace 이름, plugin 이름, owner를 내 것으로 바꾸고, 내 repository로 push 해줘."
3. 이후의 수정은 3.1절의 방식으로 반영된다.

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
  "description": "코드/문서 작성 공용 규칙: coding_rules·md_doc_rules skill, 대화 규칙과 필수 skill 로딩을 주입하는 UserPromptSubmit hook.",
  "version": "0.1.0"
}
```

## 4. Common Setup

이 절은 두 interface에 공통으로 적용되는 설정을 다룬다. 적는 내용은 하나이며, 그 내용을 어느 settings file에 두는가에 따라 적용 범위가 달라진다. 기준이 되는 위치는 project settings이고, 범위별 적용 여부는 3.1절의 표에 정리되어 있다.

### 4.1 Project Settings Declaration

project settings는 project 안의 `.claude/settings.json` 이다. 사용자의 machine이 아니라 project에 속하므로 commit 되어 함께 배포되며, 그 project를 여는 모든 사용자와 두 interface에 동일하게 적용된다. 그래서 이 file 하나가 공통 설정의 기준이 된다.

여기에 marketplace 위치와 plugin 활성화를 적는다.

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

**`"autoUpdate": true` 가 automatic의 핵심이다.** 이 한 줄이 session 시작 시 marketplace와 plugin을 GitHub 최신 기준으로 자동으로 내려받게 한다. 나머지 field는 무엇을 받을지 가리킬 뿐이고, 저절로 갱신되게 만드는 것은 이 값이다.

project를 처음 여는 사용자에게는 marketplace 설치 여부를 묻는 확인 절차가 나타난다. 신뢰하지 않는 project의 plugin이 임의로 설치되지 않도록 하는 장치이다.

### 4.2 Field Reference

| Field | Role |
|---|---|
| `extraKnownMarketplaces` | marketplace의 이름과 source를 등록한다 |
| `source.source` | source type을 지정하며 `github`, `git`, `url`, `npm`, `file`, `directory` 를 지원한다 |
| `autoUpdate` | session 시작 시 marketplace와 plugin을 자동으로 갱신한다 |
| `enabledPlugins` | `plugin-name@marketplace-name` 형식의 key를 `true`로 두어 활성화한다 |

field의 이름과 의미는 어느 settings file에 두든 동일하다.

### 4.3 Settings Precedence

settings file은 여러 층으로 나뉘며, 우선순위는 `user < project` 순이다. 그 위에 실행할 때 지정하는 option과 OS 단위로 배포하는 managed settings가 있다.

| Settings file | Location | Desktop interface | Web interface | Applies to |
|---|---|---|---|---|
| `~/.claude/settings.json` | home folder | 적용 | 미적용 | 해당 machine의 모든 project |
| `<project>/.claude/settings.json` | git repository | 적용 | 적용 | 해당 project |

`~/.claude/settings.json` 은 사용자가 자기 machine에 두는 file이므로 Web interface에는 그 file 자체가 없다.

OS 단위로 배포하는 managed settings file은 Desktop interface에만 도달하고 Web interface에는 적용되지 않는다.

## 5. Update Timing

### 5.1 Session-Level Update

`autoUpdate`를 `true`로 두면 Claude Code가 session 시작 시 `[1]`을 기준으로 사본을 갱신한다.

| Aspect | Desktop interface | Web interface |
|---|---|---|
| 갱신 시점 | session 시작 | session 시작 |
| `autoUpdate: true`의 역할 | 남아 있는 사본을 갱신한다 | 갱신할 이전 사본이 없어 무의미하다 |
| 결과 | 최신 commit | 최신 commit |

### 5.2 Prompt-Level Injection

marketplace 갱신이 session 단위인 것과 달리, UserPromptSubmit hook은 prompt 단위로 동작한다. 이 hook이 repository를 다시 읽는 것은 아니며, 이미 내려받아 둔 사본의 내용을 매 prompt마다 context에 넣는다.

| Aspect | Desktop interface | Web interface |
|---|---|---|
| 정의 위치 | plugin의 `hooks/` | 동일 |
| 실행 시점 | 매 prompt | 매 prompt |
| 읽는 대상 | `[3]` 설치된 plugin | `[4]` 설치된 plugin |
| `[1]` 재조회 | 없다 | 없다 |

따라서 rule을 push 한 뒤 열려 있는 session에서 prompt를 반복해도 새 내용은 들어오지 않는다. 두 interface 모두 새 session을 열어야 반영된다.

## 6. Verification

### 6.1 Session Command

session 안에서 `/plugin`을 입력하면 설치된 marketplace와 plugin 목록이 나타난다. 두 interface에서 모두 사용할 수 있다.

### 6.2 Claude CLI

Desktop interface에서는 `claude` CLI로도 확인할 수 있다. shell 종류와 무관하므로 PowerShell, bash 등 아무 terminal에서나 실행한다. 단, CLI는 별도 설치가 필요하다. 설치 방법은 [Appendix B](#appendix-b-claude-cli-installation) 를 본다.

```
claude plugin marketplace list
claude plugin list
claude plugin details yrocket-rules@claude-configuration
```

`details` 결과에 skill과 hook이 나타나면 정상이다.

### 6.3 Plugin Folder

Desktop interface에서만 가능하다. `~/.claude/plugins/` 아래에 marketplace clone과 plugin 사본이 생겼는지 확인한다. 수정하지 않고 보기만 한다.

## 7. Extension

repository에는 Claude Code가 읽는 경로가 정해져 있다. 그 밖의 file과 folder는 무시되므로 자유롭게 추가할 수 있다. 이 절의 확장이 놓이는 자리는 다음과 같다.

```
<repository root>/
├── .claude-plugin/
│   └── marketplace.json          : plugin 추가 시 등록
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

### 7.1 Hook

`hooks/` 에 rule file을 추가하고 `hooks.json`에 event를 연결한다. UserPromptSubmit에 묶으면 매 prompt마다 조건 없이 context에 들어간다.

### 7.2 Command

`commands/` 에 md file을 추가하면 file 이름이 slash command 이름이 된다. 사용자가 호출할 때만 load 된다.

### 7.3 Agent

`agents/` 에 md file을 추가한다. subagent의 정의이며, 호출될 때만 load 된다.

### 7.4 Plugin

`plugins/` 아래에 새 plugin folder를 만들고 `.claude-plugin/plugin.json`을 둔다. `marketplace.json` 등록이 필요한 유일한 확장이다.

### 7.5 Skill and Reference File

`skills/<skill-name>/SKILL.md` 를 추가한다. folder 이름이 skill 이름이 되고, 상세 내용은 `references/` 로 분리한다. 작성 방법은 [Appendix C](#appendix-c-skill) 를 본다.

### 7.6 Non-Loaded Document

설계 memo나 참고 자료처럼 Claude Code가 읽을 필요가 없는 문서는 plugin 구조 밖에 둔다. `docs/` 같은 folder는 무시되므로 동작에 영향을 주지 않는다.

repository 전체가 실행용 사본으로 복사되므로 용량이 큰 file은 피한다. 필요하면 4.1절의 `.claude/settings.json`에 있는 `source` object에 `"sparsePaths": [<path>, ...]` 를 지정하여 일부 folder만 받도록 제한할 수 있다.

## 8. Constraints

- plugin에 담을 수 있는 component는 skill, hook, command, agent뿐이다. CLAUDE.md는 plugin에 담을 수 없으므로, 반드시 지켜야 할 지시는 매 prompt 주입되는 UserPromptSubmit hook에 둔다.
- 저장된 CLAUDE.md를 치환하는 것은, hook이 임의의 command를 실행할 수 있으므로, SessionStart hook에 "cache의 CLAUDE.md를 project로 복사"를 시켜서 기술적으로 가능하다. 하지만 CLAUDE.md는 session 시작 시 읽히는데, hook도 session 시작 시 돌므로 복사 결과가 이번 session에 잡힌다는 보장이 없다.
- private repository는 GitHub 인증이 된 환경에서만 설치된다. 설치가 실패하면 repository 공개 범위를 확인한다.

## Appendix A. Terminology

- **`autoUpdate`**: marketplace 등록 항목의 field이다. session 시작 시 marketplace와 plugin을 remote 기준으로 갱신한다.

- **claude.ai**: Claude 계정으로 접속하는 web service이다. Web interface는 이 service 안에서 열리며, 개인 skill을 켜고 끄는 화면과 조직의 server-managed settings를 다루는 admin 화면도 여기에 있다. 두 interface가 코드를 다루는 곳이라면, claude.ai는 계정과 조직을 다루는 곳이다.

- **Claude Code in the Desktop**: local machine에서 실행하는 interface이다. `~/.claude/` 아래의 global settings와 plugin cache를 사용한다. Desktop interface로 줄여 쓴다.

- **Claude Code on the Web**: cloud에서 repository를 clone 하여 실행하는 interface이다. local file system이 없으므로 repository에 commit 된 설정만 적용된다. Web interface로 줄여 쓴다.

- **Context**: model이 응답을 만들 때 참조하는 입력 전체이다. 대화 내용과 함께 load 된 skill, hook이 주입한 rule이 여기에 들어간다. 크기에 한계가 있으므로 항상 load 되는 file은 짧게 유지한다.

- **Global settings**: `~/.claude/settings.json` 이다. 사용자가 자기 machine에 두는 file이므로 그 machine의 모든 project에 적용되고, Web interface에는 존재하지 않는다.

- **Hook**: 지정한 event에 개입하는 실행 지점이다. session 시작 시 동작하는 SessionStart hook과 매 prompt마다 동작하는 UserPromptSubmit hook을 사용한다.

- **Marketplace**: plugin의 catalog이다. `.claude-plugin/marketplace.json` 하나로 정의하며, 어떤 plugin이 어디에 있는지 나열한다.

- **Plugin**: 배포 단위이다. `.claude-plugin/plugin.json`에 자신의 이름과 정보를 두고, 하위 folder에 skill, hook, command, agent를 담는다.

- **Plugin cache**: Desktop interface가 내려받은 plugin을 실행하는 위치인 `~/.claude/plugins/cache/` 이다. 원본을 그 자리에서 쓰지 않고 이곳으로 복사해 실행한다. Claude Code가 관리하므로 직접 수정하지 않는다.

- **Project in Claude Code**: 작업 중인 directory 그 자체이며, 보통 git repository이다. 별도로 등록하는 절차가 없고, Claude Code를 연 위치가 곧 project가 된다. `.claude/` 와 CLAUDE.md도 그 directory를 기준으로 찾는다. Web interface에서는 clone 된 repository가 project가 된다.

- **Project settings**: project의 `.claude/settings.json` 이다. commit 되므로 두 interface와 다른 사용자에게 모두 적용된다.

- **Prompt**: session 안에서 사용자가 보내는 한 번의 입력이다. session보다 작은 단위이며, UserPromptSubmit hook은 이 단위로 동작한다.

- **Server-managed settings**: 조직 단위로 배포되는 settings이다. Team이나 Enterprise plan의 조직에서 owner 또는 admin이 claude.ai의 admin 화면에서 설정하며, session 시작 시 서버에서 내려와 두 interface에 모두 적용된다. OS 단위로 배포하는 managed settings file과 달리 Web interface까지 도달한다.

- **Session**: 하나의 Claude Code 실행 단위이다. 시작 시점에 settings, plugin, CLAUDE.md, hook 정의를 읽어 들이고, 그 구성으로 대화가 끝날 때까지 동작한다. rule 변경이 반영되는 경계가 곧 session이다.

- **Skill**: 필요한 시점에 load 되는 rule 묶음이다. folder 이름이 skill 이름이 되고, `SKILL.md`의 `description`이 load 시점을 결정한다.

- **Working clone**: rule을 수정하기 위해 받아 둔 marketplace repository의 clone이다. 실행에는 쓰이지 않으며, push를 통해서만 동작에 반영된다.

## Appendix B. Claude CLI Installation

Desktop app을 설치해도 `claude` CLI는 PATH에 들어오지 않는다. terminal에서 `claude` 명령을 쓰려면 별도로 설치한다.

| Environment | Command |
|---|---|
| Windows (PowerShell) | `irm https://claude.ai/install.ps1 \| iex` |
| macOS / Linux / WSL | `curl -fsSL https://claude.ai/install.sh \| bash` |

WSL은 Windows와 별개의 환경이므로 양쪽에서 쓰려면 각각 설치한다. 설치 후 terminal을 새로 열고 `claude --version` 으로 설치를 확인한다.

## Appendix C. Skill

### C.1 Skill

새 rule 묶음은 skill folder를 추가하는 것으로 끝난다. plugin manifest는 수정하지 않아도 된다.

file 이름은 반드시 `SKILL.md`이며, folder 이름이 skill 이름이 된다. frontmatter의 `description`이 언제 이 skill을 load 할지 판단하는 근거이므로, 적용 시점을 분명히 적는다.

사용자가 prompt에 `/<skill-name>` 을 입력하면 `description`의 판단과 무관하게 그 skill이 바로 load 된다.

```markdown
---
name: <skill-name>
description: 적용 대상과 load 시점을 명시한다.
---
```

### C.2 Reference File

skill의 상세 내용은 별도 file로 분리하고 SKILL.md에서 가리킨다. SKILL.md는 항상 load 되지만 reference file은 필요할 때만 읽히므로, 기본 context 소비를 줄이면서 깊은 내용에 도달할 수 있다.

```
skills/<skill-name>/
├── SKILL.md              (always loaded, keep short)
└── references/
    └── <topic>.md        (loaded on demand)
```

## Appendix D. Obsidian

Obsidian은 local folder에 놓인 md file을 그대로 읽고 쓰는 편집 도구이다. 자체 file 형식이나 database를 두지 않고, folder 하나를 vault로 지정해 그 안의 md file을 note로 다룬다. file 사이의 link를 따라가거나 전체를 한 번에 검색하는 기능이 편집기와 다른 점이다.

rule은 결국 md file이므로 Obsidian으로 편집할 수 있다. vault에 별도의 형식이 없으므로 working clone을 vault로 삼으면 그대로 열린다.

### D.1 Vault Placement

배치는 세 가지가 있다.

| Placement | Description | Note |
|---|---|---|
| working clone 자체를 vault로 연다 | rule만 담긴 독립 vault가 된다 | 가장 단순하며 다른 기록과 섞이지 않는다 |
| 기존 vault 안에 working clone을 둔다 | vault 하위 folder가 git repository가 된다 | 기존 note와 함께 검색되지만, vault 전체를 sync 하는 도구와 git이 같은 file을 건드린다 |
| vault 밖에 두고 symlink를 건다 | 실체는 vault 밖에 있고 vault에는 link만 둔다 | 배치는 자유롭지만 Obsidian이 link 대상을 vault 경계 밖으로 인식하는 경우가 있다 |

기록용 vault를 이미 쓰고 있다면 첫 번째를 권한다. rule repository는 push 시점이 곧 배포 시점이라, 일반 note와 commit 주기를 섞지 않는 편이 안전하다.

### D.2 Link Style

Obsidian의 기본 link 형식인 `[[wikilink]]` 는 Claude Code가 해석하지 않는다. skill이 reference file을 가리키는 link는 상대 경로 markdown link로 적는다.

```markdown
좋음: 자세한 내용은 [naming](references/naming.md) 을 본다.
나쁨: 자세한 내용은 [[naming]] 을 본다.
```

Obsidian 설정에서 wikilink를 끄고 link를 상대 경로로 두면, 새로 만드는 link도 같은 형식이 된다. vault 안에서만 쓰는 note끼리는 wikilink를 써도 무방하지만, plugin folder 아래 file에는 쓰지 않는다.

### D.3 File Hygiene

working clone 자체를 vault로 열었을 때의 구조는 다음과 같다. Obsidian이 더하는 것은 최상위의 `.obsidian/` 하나뿐이다.

```
<working clone> = vault root
├── .obsidian/                : Obsidian settings, Claude Code는 읽지 않는다
│   ├── app.json              : editor 설정, link 형식을 여기서 정한다
│   ├── workspace.json        : 창 배치, 자주 바뀐다
│   └── plugins/              : Obsidian community plugin
├── .claude-plugin/
│   └── marketplace.json
├── plugins/                  : Claude Code plugin, 첨부 file을 두지 않는다
│   └── <plugin-name>/
│       ├── .claude-plugin/
│       │   └── plugin.json
│       └── skills/
│           └── <skill-name>/
│               ├── SKILL.md
│               └── references/
└── docs/
    ├── <document>.md
    └── assets/               : 첨부 file 위치를 여기로 지정한다
```

`plugins` 라는 이름이 두 곳에 나오지만 서로 관계가 없다. `.obsidian/plugins/` 는 Obsidian이 쓰는 것이고, 최상위의 `plugins/` 는 Claude Code가 쓰는 것이다.

`.obsidian/` 은 혼자 쓰는 vault라면 `.gitignore` 에 넣고, 설정을 함께 쓰고 싶다면 commit 한다.

첨부 file이 생길 자리를 미리 지정한다. 지정하지 않으면 note를 만든 위치, 즉 plugin folder 안에 image가 쌓인다.

frontmatter를 자동으로 정리하는 부류의 community plugin은 SKILL.md에 적용하지 않는다. `name` 과 `description` 은 skill이 언제 load 될지 정하는 값이라, key 순서나 표기가 바뀌면 의도와 다르게 동작할 수 있다.

### D.4 Sync

vault 전체를 장치 사이에서 자동으로 복제하는 기능이 있다. Obsidian이 제공하는 sync service, vault folder를 cloud drive 안에 두는 방식, folder를 실시간으로 맞추는 file 동기화 도구가 모두 여기에 해당한다. 이 기능을 모두 끄고 Obsidian은 편집기로만 사용하며, rule repository는 git의 commit과 push로만 관리하여 원본이 remote repository에 있도록 한다.
