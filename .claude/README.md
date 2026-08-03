# Claude Rules Setup
rev. 1

이 폴더는 세션이 시작될 때마다 공용 규칙이 자동으로 실리도록 구성되어 있다. 다른 repo 에도 같은 방식을 그대로 적용할 수 있다.

## 1. Constraint

plugin 은 **세션이 시작되는 시점에 이미 설치되어 있어야만** 실린다. SessionStart hook 이 plugin 을 설치해도 그 세션에서는 잡히지 않고 다음 세션부터 잡힌다. Claude Code on the web 은 매번 새 container 에서 시작하므로 항상 첫 번째 경우에 해당하고, 따라서 plugin 방식으로는 규칙이 실리지 않는다.

`.claude/skills/` 에 둔 standalone skill 은 install 단계 없이 checkout 에서 바로 읽히므로 이 문제가 없다. 이름에 namespace 도 붙지 않아 `/md_rules` 로 호출한다.

## 2. Layout

Table 1. Files and their roles

| Path | Role |
|---|---|
| `.claude/skills/<name>/SKILL.md` | `/<name>` 으로 호출하는 skill 이다. 폴더 이름이 곧 호출 이름이다. |
| `.claude/hooks/conversation_rules.md` | 매 prompt 마다 주입할 대화 규칙이다. |
| `.claude/hooks/skill_loading_rules.md` | 매 prompt 마다 주입할 skill 로딩 규칙이다. |
| `.claude/settings.json` | 위 두 파일을 주입하는 UserPromptSubmit hook 을 등록한다. |
| `.claude/scripts/sync-yrocket-plugins.sh` | 원본 repo 에서 skill 과 규칙 파일을 갱신한다. |

## 3. Setup

### 3.1 Copy the files

대상 repo 의 root 에서 아래를 실행하여 skill 과 규칙 파일을 가져온다. `<SOURCE_REPO>` 는 원본 checkout 경로이다.

```bash
# run from the target repository root
mkdir -p .claude/skills .claude/hooks
cp -a <SOURCE_REPO>/plugins/yrocket-plugins/skills/md_rules .claude/skills/
cp -a <SOURCE_REPO>/plugins/yrocket-plugins/skills/coding_rules .claude/skills/
cp -a <SOURCE_REPO>/plugins/yrocket-plugins/hooks/conversation_rules.md .claude/hooks/
cp -a <SOURCE_REPO>/plugins/yrocket-plugins/hooks/skill_loading_rules.md .claude/hooks/
```

### 3.2 Register the hook

`.claude/settings.json` 을 아래 내용으로 만든다. `$CLAUDE_PROJECT_DIR` 를 쓰므로 machine 마다 경로를 고칠 필요가 없다.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cat \"$CLAUDE_PROJECT_DIR/.claude/hooks/conversation_rules.md\" \"$CLAUDE_PROJECT_DIR/.claude/hooks/skill_loading_rules.md\""
          }
        ]
      }
    ]
  }
}
```

### 3.3 Commit and verify

`.claude/` 를 commit 하고 push 한다. 새 세션에서 `/md_rules` 를 호출했을 때 skill 이 실리고, 응답이 대화 규칙을 따르면 hook 까지 정상이다.

## 4. Sync

원본 repo 의 규칙이 바뀌면 `.claude/scripts/sync-yrocket-plugins.sh` 를 실행하여 사본을 갱신한다. 인자 없이 실행하면 원본을 임시 folder 에 clone 하고, 이미 checkout 이 있으면 그 경로를 인자로 넘긴다.

```bash
# refresh the vendored copy and show what changed
.claude/scripts/sync-yrocket-plugins.sh [SOURCE_FOLDER]
```

이 script 는 commit 하지 않고 바뀐 파일만 보여준다. 원본에서 파일 이름이 바뀌거나 사라지면 아무것도 덮어쓰지 않고 실패하므로, 사본이 조용히 낡은 상태로 남지 않는다.

## 5. Caution

같은 규칙을 담은 plugin 을 별도로 설치해 두면 skill 이 두 벌 잡히고 대화 규칙도 두 번 주입된다. 이 방식을 쓰는 repo 에서는 그 plugin 을 비활성화한다.

## Appendix A. Terminology

- **container**: 세션이 실행되는 격리된 실행 환경이다. Claude Code on the web 은 세션마다 새로 만든다.
- **hook**: 정해진 시점에 Claude Code 가 실행하는 command 이다. UserPromptSubmit hook 의 출력은 prompt 마다 context 에 주입된다.
- **skill**: `SKILL.md` 한 개로 정의하는 지시문 묶음이다. `/<name>` 으로 직접 호출하거나 Claude 가 필요할 때 스스로 로드한다.
- **standalone**: skill 과 hook 을 plugin 으로 배포하지 않고 repo 의 `.claude/` 에 직접 두는 방식이다.
