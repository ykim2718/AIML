# Automatic Rule Loading via Plugin Marketplace

rev. 24

## 1. Goal

Automatic rule loading은 rule을 한 repository에서 관리하고, 각 project는 그 repository를 한 번 선언해 두는 것으로 끝내는 방식이다. 선언 이후에는 두 가지가 저절로 이루어진다.

1. plugin이 새 session 시작 시 최신 상태로 update 된다.
2. rule이 새 session과 새 prompt에 적용된다.

문서에서 자동이라고 할 때는 이 두 가지를 뜻한다.

이 문서는 Claude Code in the Desktop (Desktop interface) 과 Claude Code on the Web (Web interface) 에서 동일한 plugin (skills, hooks, commands, agents) 을 이렇게 재사용하기 위해, plugin marketplace를 구성하고 적용하는 방법을 정리한다. 4절부터 8절까지가 두 interface에 공통으로 적용되는 내용이고, 9절과 10절은 각 interface에만 해당하는 사항이다.

## 2. Problem

기존 방식에는 두 가지 문제가 있다.

1. **CLAUDE.md에 GitHub URL을 적어 두는 방식은 동작하지 않는다.** CLAUDE.md는 local file import (`@path`) 만 지원하며, 원격 URL은 단순 text로 남는다. 새 session이 시작해도 해당 repository를 clone 하거나 내용을 가져오지 않는다.

2. **project마다 skill과 hook을 복사해 두는 방식은 유지가 어렵다.** rule을 고칠 때마다 모든 project를 찾아다녀야 하고, 사본 간에 내용이 어긋난다.

plugin marketplace는 이 두 문제를 함께 해결한다. rule을 한 곳에서 관리하고, 각 project는 참조만 한다.

## 3. Architecture

### 3.1 Marketplace Layers

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

marketplace manifest는 repository 최상위의 `.claude-plugin/marketplace.json` 하나뿐이며, 어떤 plugin이 어디에 있는지만 나열한다. plugin manifest는 각 plugin folder 안의 `.claude-plugin/plugin.json`으로, 그 plugin이 자신을 선언한다. 실제 기능은 plugin folder 아래의 component folder가 담는다.

component 중 load 시점이 고정된 것은 hook뿐이다. skill은 `description`이 맞을 때, command와 agent는 호출될 때 load 되지만, `hooks.json`에 UserPromptSubmit으로 묶은 file은 매 prompt마다 조건 없이 context에 들어간다. 반드시 지켜야 할 rule을 이 경로에 두는 이유이다.

### 3.2 Copies

동일한 내용이 여러 곳에 존재하지만 역할이 서로 다르다. 실행용 사본은 interface마다 별도로 존재한다.

```
                    +----------------------------------+
                    |  [1] Remote repository           |
                    |      (source of truth)           |
                    +----------------------------------+
                       ^           |              |
               git push|     fetch |        fetch |
                       |           v              v
   +-------------------+--+  +-------------------+  +----------------------+
   | [2] Working clone    |  | [3] Plugin cache  |  | [4] Session copy     |
   |     author edits     |  | ~/.claude/plugins/|  |  cloud container     |
   |                      |  |             cache/|  |                      |
   |                      |  |  Desktop interface|  |  Web interface       |
   +----------------------+  +-------------------+  +----------------------+
```

| Copy | Location | Owner | Purpose |
|---|---|---|---|
| Remote repository | GitHub | 공유 | 원본 |
| Working clone | 임의의 workspace | 사용자 | 수정과 push |
| Plugin cache | `~/.claude/plugins/cache/` | Desktop interface | 실행, machine에 남는다 |
| Session copy | cloud container | Web interface | 실행, session과 함께 사라진다 |

`[2]`, `[3]`, `[4]`는 서로를 참조하지 않는다. 모두 `[1]`만 바라본다. Claude Code가 읽는 것은 `[3]` 또는 `[4]` 뿐이므로, working clone을 수정해도 push 전까지는 동작에 영향이 없다.

## 4. Common Setup

이 절은 두 interface에 공통으로 적용되는 설정을 다룬다. 선언의 내용은 하나이며, 그 내용을 어느 settings file에 두는가에 따라 적용 범위가 달라진다. 기준이 되는 위치는 project settings이고, 더 넓은 범위는 4.4절에서 다룬다.

### 4.1 Project Settings Declaration

project settings는 project 안의 `.claude/settings.json` 이다. 사용자의 machine이 아니라 project에 속하므로 commit 되어 함께 배포되며, 그 project를 여는 모든 사용자와 두 interface에 동일하게 적용된다. 그래서 이 file 하나가 공통 설정의 기준이 된다.

여기에 marketplace 위치와 plugin 활성화를 선언한다.

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

project를 처음 여는 사용자에게는 marketplace 설치 여부를 묻는 확인 절차가 나타난다. 신뢰하지 않는 project의 plugin 선언이 임의로 실행되지 않도록 하는 장치이다.

### 4.2 Field Reference

| Field | Role |
|---|---|
| `extraKnownMarketplaces` | marketplace의 이름과 source를 등록한다 |
| `source.source` | source type을 지정하며 `github`, `git`, `url`, `npm`, `file`, `directory` 를 지원한다 |
| `autoUpdate` | session 시작 시 marketplace와 plugin을 자동으로 갱신한다 |
| `enabledPlugins` | `plugin-name@marketplace-name` 형식의 key를 `true`로 두어 활성화한다 |

field의 이름과 의미는 어느 settings file에 두든 동일하다.

### 4.3 Settings Precedence

settings file은 여러 층으로 나뉘며, 우선순위는 `user < project < local < flag < policy` 순이다.

| Settings file | Desktop interface | Web interface | Applies to |
|---|---|---|---|
| `~/.claude/settings.json` | 적용 | 미적용 | 해당 machine의 모든 project |
| `<project>/.claude/settings.json` | 적용 | 적용 | 해당 project, commit 되어 공유된다 |
| `<project>/.claude/settings.local.json` | 적용 | 적용 | 해당 project, commit 하지 않는다 |

`~/.claude/settings.json` 은 사용자가 자기 machine에 두는 file이므로 Web interface에는 그 file 자체가 없다. 사용법은 9.1절에서 다룬다.

project가 켠 plugin을 끄려면 상위 층인 global settings가 아니라 해당 project의 `.claude/settings.local.json`에 `false`를 넣어야 한다.

### 4.4 Rule Loading Coverage Beyond a Project

선언의 내용은 4.1절과 같고, 그 내용을 어느 file에 두는가가 적용 범위를 정한다. project settings에 두면 그 project에서만 적용되며, 더 넓은 범위를 덮으려면 같은 내용을 상위 file로 옮긴다. 어디까지 덮이는지는 interface마다 다르다.

| Project scope | Site | Desktop interface | Web interface |
|---|---|---|---|
| a project | every machine | project settings | project settings |
| every project | a machine | global settings, 9.1절 | 수단 없음 |
| every user's every project | every machine | server-managed settings | server-managed settings |

Site는 그 선언이 유효한 machine의 범위이다. project settings와 server-managed settings는 machine에 매이지 않으므로 어느 machine에서 열어도 유효하지만, global settings는 그 file이 놓인 machine에서만 유효하다. machine이 여러 대라면 각 machine에 따로 넣어야 한다.

Web interface의 두 번째 행이 비어 있는 것은 Web interface에 machine layer가 없기 때문이다. 선언이 없는 project를 열면 rule은 올라오지 않는다.

앞의 두 행, 즉 개인 사용자가 쓸 수 있는 범위에서는 두 interface를 한 번에 덮는 수단이 없다. Web interface에는 9.1절의 global settings에 해당하는 층이 없으므로, 적용하려는 project마다 project settings를 넣어야 한다. claude.ai에서 켠 개인 skill이 Web interface의 session에 들어오기는 하지만, 이는 skill일 뿐 plugin과 marketplace가 아니다.

마지막 행의 조직은 Team이나 Enterprise plan에서 구성원을 묶는 단위이며, 개인 plan에는 해당하지 않는다. 적용 대상은 그 조직에 속한 사용자 전체이고, 설정은 owner 또는 admin 권한을 가진 사람이 claude.ai의 admin 화면에서 바꾼다. 다만 `enabledPlugins`로 특정 plugin을 강제하는 것은 가능하나 `extraKnownMarketplaces`를 이 경로로 배포하는 방법은 문서에 없으므로, marketplace 등록은 여전히 project settings가 맡는다. OS 단위로 배포하는 managed settings file은 Desktop interface에만 도달하고 Web interface에는 적용되지 않는다.

조직 수단을 쓸 수 없는 개인 사용자, 즉 앞의 두 행 기준으로 정리하면 다음과 같다. 두 interface 공통 기준으로 project settings를 두고, Desktop interface의 편의를 위해 global settings를 함께 둔다.

자동의 의미는 1절과 같다. 선언이 닿는 범위 안에서 plugin이 자동으로 update 되고 새 session과 새 prompt에 자동으로 적용된다. 그 범위는 Desktop interface에서는 machine 전체이고, Web interface에서는 선언이 들어 있는 project이다.

## 5. Update Timing

### 5.1 Session-Level Update

`autoUpdate`를 `true`로 두면 별도의 동기화 장치가 필요하지 않다. Claude Code가 session 시작 시 marketplace manifest와 설치된 plugin을 갱신한다. marketplace가 아닌 일반 file 묶음이었다면 SessionStart hook이나 OS scheduler로 pull을 걸어야 하지만, plugin 방식에서는 그 계층이 불필요하다.

두 interface 모두 같은 GitHub repository를 원본으로 읽는다. Web interface 전용 marketplace 사본이 별도로 존재하는 것이 아니다.

| Aspect | Desktop interface | Web interface |
|---|---|---|
| Source of truth | GitHub repository | 동일한 GitHub repository |
| Fetch 주체 | local machine의 Claude Code | cloud environment의 Claude Code |
| Fetch 대상 위치 | `~/.claude/plugins/cache/` (machine에 남는다) | 해당 session의 container (session과 함께 사라진다) |
| Fetch 시점 | session 시작 | session 시작 |
| `autoUpdate` 반영 시점 | session 시작 (new prompt에서는 동작하지 않는다) | session 시작 (매번 새로 설치되므로 시점이 곧 설치 시점이다) |
| `autoUpdate: true`의 역할 | 남아 있는 cache를 remote 기준으로 갱신한다 | 갱신할 이전 cache가 없어 사실상 무의미하다 |
| 결과 | 최신 commit | 최신 commit |

차이는 무엇을 읽는가가 아니라 이전 사본이 남아 있는가이다. Desktop interface는 사본이 machine에 계속 남으므로 remote와 맞추는 동작이 필요하고, Web interface는 빈 상태에서 시작하므로 갱신이라는 단계 자체가 없다. 도달하는 commit은 양쪽 모두 marketplace repository의 최신 commit이다.

### 5.2 Prompt-Level Injection

marketplace 갱신이 session 단위인 것과 달리, UserPromptSubmit hook은 prompt 단위로 동작한다. 이 hook이 repository를 다시 읽는 것은 아니며, 이미 내려받아 둔 사본의 내용을 매 prompt마다 context에 넣는다.

| Aspect | Desktop interface | Web interface |
|---|---|---|
| Source of truth | GitHub repository | 동일한 GitHub repository |
| 정의 위치 | plugin의 `hooks/` | 동일 |
| 실행 시점 | 매 prompt | 매 prompt |
| 읽는 대상 | `~/.claude/plugins/cache/` 의 사본 | session container 안의 사본 |
| Repository 재조회 | 없다 | 없다 |
| 수정 반영 조건 | 새 session | 새 session |

따라서 rule을 push 한 뒤 열려 있는 session에서 prompt를 반복해도 새 내용은 들어오지 않는다. 두 interface 모두 새 session을 열어야 반영된다.

## 6. Working Clone Separation

rule을 수정하려면 working clone이 필요하다. 실행용 사본은 auto update가 덮어쓰므로 편집 대상이 아니다.

working clone은 실행용 사본과 다른 위치에 두는 것이 좋다. 같은 영역에 두면 실행용 사본과 편집용 사본을 혼동하기 쉽다. 일반 workspace folder에 repository 이름으로 clone 한다.

수정 흐름은 다음과 같다.

```
edit working clone -> commit -> push
                                  |
                    next session start: autoUpdate pulls into cache
                                  |
                              rule applied
```

local folder를 `directory` source로 등록하면 working clone도 marketplace로 참여하게 된다. 이때는 remote marketplace와 skill 이름이 겹쳐 충돌할 수 있으므로, 한쪽을 비활성화한 상태로 사용한다.

## 7. Verification

session 안에서는 `/plugin`으로 설치 상태를 확인한다. 두 interface에서 모두 사용할 수 있다.

Desktop interface에서는 `claude` CLI로도 확인할 수 있다. shell 종류와 무관하므로 PowerShell, bash 등 아무 terminal에서나 실행한다.

```
claude plugin marketplace list
claude plugin list
claude plugin details yrocket-rules@claude-configuration
```

`details` 결과에 skill과 hook이 나타나면 정상이다.

## 8. Extension

repository에는 Claude Code가 읽는 경로가 정해져 있다. 그 밖의 file과 folder는 무시되므로 자유롭게 추가할 수 있다.

### 8.1 Reference File

skill의 상세 내용은 별도 file로 분리하고 SKILL.md에서 가리킨다. SKILL.md는 항상 load 되지만 reference file은 필요할 때만 읽히므로, 기본 context 소비를 줄이면서 깊은 내용에 도달할 수 있다.

```
skills/<skill-name>/
├── SKILL.md              (always loaded, keep short)
└── references/
    └── <topic>.md        (loaded on demand)
```

### 8.2 New Skill

새 rule 묶음은 skill folder를 추가하는 것으로 끝난다. plugin manifest는 수정하지 않아도 된다.

file 이름은 반드시 `SKILL.md`이며, folder 이름이 skill 이름이 된다. frontmatter의 `description`이 언제 이 skill을 load 할지 판단하는 근거이므로, 적용 시점을 분명히 적는다.

```markdown
---
name: <skill-name>
description: 적용 대상과 load 시점을 명시한다.
---
```

### 8.3 Non-Loaded Document

설계 memo나 참고 자료처럼 Claude Code가 읽을 필요가 없는 문서는 plugin 구조 밖에 둔다. `docs/` 같은 folder는 무시되므로 동작에 영향을 주지 않는다.

repository 전체가 실행용 사본으로 복사되므로 용량이 큰 file은 피한다. 필요하면 marketplace source에 `sparsePaths`를 지정하여 일부 folder만 받도록 제한할 수 있다.

## 9. Desktop Interface Specifics

### 9.1 Global Settings

Desktop interface에는 machine 단위 settings file인 `~/.claude/settings.json` 이 있다. 내용의 형식은 4.1절과 같으며, 그 machine에서 여는 모든 project에 적용된다.

아직 rule을 넣지 않은 project까지 덮고 싶을 때 사용한다. 두 interface에 같은 rule을 적용하는 것이 목적이라면 project settings가 기준이고, global settings는 machine 단위 fallback으로만 남긴다.

### 9.2 Plugin Cache

Desktop interface는 내려받은 plugin을 `~/.claude/plugins/cache/` 에 복사해 두고 실행한다. 원본을 그 자리에서 쓰지 않고 옮겨 쓰는 구조이며, 이 folder는 session이 끝나도 남는다.

plugin cache를 직접 수정하면 안 된다. auto update가 remote 기준으로 덮어쓰므로 수정 내용이 사라진다. cache folder가 비어 있다면 아직 session이 시작되지 않은 상태이다.

## 10. Web Interface Specifics

Web interface는 cloud에서 repository를 clone 하여 session을 연다. local machine의 file system이 존재하지 않으므로 `~/.claude/settings.json`은 읽히지 않는다.

### 10.1 Availability

session에 들어가는 것은 clone 된 repository의 내용뿐이다.

| Location | Web interface | Note |
|---|---|---|
| `~/.claude/settings.json` | 미적용 | machine 안에만 존재한다 |
| `~/.claude/CLAUDE.md`, `~/.claude/skills/` | 미적용 | 같은 이유로 Web interface에 없다 |
| `<project>/CLAUDE.md` | 적용 | clone에 포함된다 |
| `<project>/.claude/settings.json` | 적용 | plugin 선언과 hook이 모두 동작한다 |
| `<project>/.claude/skills/`, `agents/`, `commands/` | 적용 | clone에 포함된다 |

plugin marketplace 자체는 Web interface에서도 지원된다. 4.1절의 선언은 그대로 동작하며, plugin은 session 시작 시 marketplace에서 설치된다.

### 10.2 Network Access

cloud environment는 outbound 접근 수준을 `None`, `Trusted`, `Full`, `Custom` 중 하나로 갖는다. 기본값인 `Trusted`의 allowlist에 GitHub domain이 포함되어 있으므로, GitHub source marketplace는 추가 설정 없이 설치된다.

접근 수준을 `None`으로 두면 marketplace를 받아오지 못해 plugin이 없는 상태로 session이 시작한다. GitHub 이외의 host에 marketplace를 둔 경우에는 `Custom`으로 해당 domain을 열어야 한다.

### 10.3 Session Start Work

session 시작 시 실행할 작업은 두 가지 경로가 있다.

| Mechanism | Defined in | Timing | Repeat |
|---|---|---|---|
| Setup script | cloud environment 설정 화면 | Claude Code 기동 전 | 최초 1회, 결과를 cache |
| SessionStart hook | project의 `.claude/settings.json` | Claude Code 기동 후 | 매 session |

rule loading은 plugin이 담당하므로 둘 다 필수는 아니다. Web interface에서만 필요한 dependency 설치는 setup script에, 두 interface에 공통으로 필요한 준비 작업은 SessionStart hook에 둔다. hook 안에서 `CLAUDE_CODE_REMOTE` environment variable을 확인하면 한쪽 interface에서만 건너뛰게 만들 수 있다.

## 11. Constraints

plugin은 CLAUDE.md를 주입하지 못한다. 그래서 반드시 지켜야 할 skill loading 지시는 UserPromptSubmit hook으로 전달한다. 이 hook은 매 prompt마다 rule을 context에 넣는다.

plugin은 hook을 통해 임의의 command를 실행할 수 있다. 외부 marketplace를 등록할 때는 hook 정의를 먼저 확인한다.

private repository는 GitHub 인증이 된 환경에서 정상적으로 설치된다. 설치가 실패하면 repository 공개 범위를 확인한다.

## Appendix A. Terminology

- **Session**: 하나의 Claude Code 실행 단위이다. 문서에서는 Claude session이라고도 적는다. 시작 시점에 settings, plugin, CLAUDE.md, hook 정의를 읽어 들이고, 그 구성으로 대화가 끝날 때까지 동작한다. rule 변경이 반영되는 경계가 곧 session이므로, 이미 열려 있는 session은 prompt를 아무리 반복해도 새 rule을 받지 않는다.

- **Prompt**: session 안에서 사용자가 보내는 한 번의 입력이다. session보다 작은 단위이며, UserPromptSubmit hook은 이 단위로 동작한다.

- **claude.ai**: Claude 계정으로 접속하는 web service이다. Web interface는 이 service 안에서 열리며, 개인 skill을 켜고 끄는 화면과 조직의 server-managed settings를 다루는 admin 화면도 여기에 있다. 두 interface가 코드를 다루는 곳이라면, claude.ai는 계정과 조직을 다루는 곳이다.

- **Claude Code in the Desktop**: local machine에서 실행하는 interface이다. `~/.claude/` 아래의 global settings와 plugin cache를 사용한다. Desktop interface로 줄여 쓴다.

- **Claude Code on the Web**: cloud에서 repository를 clone 하여 실행하는 interface이다. local file system이 없으므로 repository에 commit 된 설정만 적용된다. Web interface로 줄여 쓴다.

- **Marketplace**: plugin의 catalog이다. `.claude-plugin/marketplace.json` 하나로 정의하며, 어떤 plugin이 어디에 있는지 나열한다.

- **Plugin**: 배포 단위이다. `.claude-plugin/plugin.json`으로 자신을 선언하고, 하위 folder에 skill, hook, command, agent를 담는다.

- **Skill**: 필요한 시점에 load 되는 rule 묶음이다. folder 이름이 skill 이름이 되고, `SKILL.md`의 `description`이 load 시점을 결정한다.

- **Hook**: 지정한 event에 개입하는 실행 지점이다. session 시작 시 동작하는 SessionStart hook과 매 prompt마다 동작하는 UserPromptSubmit hook을 사용한다.

- **Plugin cache**: Desktop interface가 marketplace의 plugin을 복사해 두고 실행하는 위치인 `~/.claude/plugins/cache/` 이다. 원본을 그 자리에서 쓰지 않고 이곳으로 옮겨 실행한다. Claude Code가 관리하므로 직접 수정하지 않는다. 같은 상위 folder의 `data/` 는 plugin이 update 되어도 남는 영역이라 성격이 다르다.

- **Session copy**: Web interface가 session container 안에 내려받는 실행용 사본이다. session이 끝나면 사라진다.

- **Working clone**: rule을 수정하기 위해 사용자가 별도 workspace에 둔 clone이다. 실행에는 쓰이지 않으며, push를 통해서만 동작에 반영된다.

- **Project in Claude Code**: 작업 중인 directory 그 자체이며, 보통 git repository이다. 별도로 등록하거나 선언하는 절차가 없고, Claude Code를 연 위치가 곧 project가 된다. `.claude/` 와 CLAUDE.md도 그 directory를 기준으로 찾는다. Web interface에서는 clone 된 repository가 project가 된다.

- **Global settings**: `~/.claude/settings.json` 이다. 사용자가 자기 machine에 두는 file이므로 그 machine의 모든 project에 적용되고, Web interface에는 존재하지 않는다.

- **Project settings**: project의 `.claude/settings.json` 이다. commit 되므로 두 interface와 다른 사용자에게 모두 적용된다.

- **Server-managed settings**: 조직 단위로 배포되는 settings이다. Team이나 Enterprise plan의 조직에서 owner 또는 admin이 claude.ai의 admin 화면에서 설정하며, session 시작 시 서버에서 내려와 두 interface에 모두 적용된다. OS 단위로 배포하는 managed settings file과 달리 Web interface까지 도달한다.

- **`autoUpdate`**: marketplace 등록 항목의 field이다. session 시작 시 marketplace와 plugin을 remote 기준으로 갱신한다.

- **Setup script**: cloud environment 설정 화면에서 지정하는 script이다. Claude Code 기동 전에 최초 1회 실행되고 결과가 cache 된다.

## Appendix B. Obsidian

Obsidian은 local folder에 놓인 md file을 그대로 읽고 쓰는 편집 도구이다. 자체 file 형식이나 database를 두지 않고, folder 하나를 vault로 지정해 그 안의 md file을 note로 다룬다. file 사이의 link를 따라가거나 전체를 한 번에 검색하는 기능이 편집기와 다른 점이다.

rule은 결국 md file이므로 Obsidian으로 편집할 수 있다. vault에 별도의 형식이 없으므로 working clone을 vault로 삼으면 그대로 열린다. 편집 대상은 언제나 working clone이며, plugin cache는 vault에 넣지 않는다.

### B.1 Vault Placement

배치는 세 가지가 있다.

| Placement | Description | Note |
|---|---|---|
| working clone 자체를 vault로 연다 | rule만 담긴 독립 vault가 된다 | 가장 단순하며 다른 기록과 섞이지 않는다 |
| 기존 vault 안에 working clone을 둔다 | vault 하위 folder가 git repository가 된다 | 기존 note와 함께 검색되지만, vault 전체를 sync 하는 도구와 git이 같은 file을 건드린다 |
| vault 밖에 두고 symlink를 건다 | 실체는 vault 밖에 있고 vault에는 link만 둔다 | 배치는 자유롭지만 Obsidian이 link 대상을 vault 경계 밖으로 인식하는 경우가 있다 |

기록용 vault를 이미 쓰고 있다면 첫 번째를 권한다. rule repository는 push 시점이 곧 배포 시점이라, 일반 note와 commit 주기를 섞지 않는 편이 안전하다.

### B.2 Link Style

Obsidian의 기본 link 형식인 `[[wikilink]]` 는 Claude Code가 해석하지 않는다. skill이 reference file을 가리키는 link는 상대 경로 markdown link로 적는다.

```markdown
좋음: 자세한 내용은 [naming](references/naming.md) 을 본다.
나쁨: 자세한 내용은 [[naming]] 을 본다.
```

Obsidian 설정에서 wikilink를 끄고 link를 상대 경로로 두면, 새로 만드는 link도 같은 형식이 된다. vault 안에서만 쓰는 note끼리는 wikilink를 써도 무방하지만, plugin folder 아래 file에는 쓰지 않는다.

### B.3 File Hygiene

Obsidian은 vault 최상위에 `.obsidian/` 을 만들고 자신의 설정을 담는다. Claude Code는 이 folder를 읽지 않으므로 동작에는 영향이 없다. 혼자 쓰는 vault라면 `.gitignore` 에 넣고, 설정을 함께 쓰고 싶다면 commit 한다.

첨부 file이 생길 자리를 미리 지정한다. 지정하지 않으면 note를 만든 위치, 즉 plugin folder 안에 image가 쌓인다. 8.3절이 말한 대로 repository 전체가 실행용 사본으로 복사되므로 용량이 늘어난다.

frontmatter를 자동으로 정리하는 부류의 community plugin은 SKILL.md에 적용하지 않는다. `name` 과 `description` 은 skill이 언제 load 될지 정하는 값이라, key 순서나 표기가 바뀌면 의도와 다르게 동작할 수 있다.

### B.4 Sync

vault 단위 sync 기능과 git을 같은 folder에 함께 걸지 않는다. 두 도구가 각자 판단으로 file을 되돌리면 어느 쪽이 최신인지 알 수 없게 된다.

rule repository의 원본은 git이다. Obsidian은 편집기로만 쓰고, 반영은 commit과 push로 한다. 6절의 흐름이 그대로 적용된다.
