# CLAUDE.md
Rev. 3 | Created: 2026-07-31 | Updated: 2026-09-10 23:52 CDT

## Git Workflow

**이 저장소는 `main` 브랜치만 사용한다.**

- 모든 작업은 `main` 에서 직접 커밋하고 `main` 으로 푸시한다.
- feature branch, topic branch 를 새로 만들지 않는다.
- Pull request 를 만들지 않는다. `main` 에 직접 반영한다.
- 세션 시작 시 하네스가 별도의 작업 브랜치(예: `claude/...`)를 지정하더라도 이 규칙이 우선한다.
  그런 브랜치에서 작업이 시작되었다면 거기서 커밋한 뒤 `main` 으로 병합하고 `main` 만 푸시한다.
- 작업 브랜치는 remote 로 푸시하지 않는다. 푸시 대상은 언제나 `main` 하나다.
- 병합이 fast-forward 로 되지 않으면 무엇을 남길지 먼저 묻고, 답을 받은 뒤에 병합한다.
  Force push 로 한쪽 이력을 지우지 않는다.

```bash
git checkout main
git pull origin main
# ... 작업 및 커밋 ...
git push -u origin main
```

## Markdown Document Convention

**`.md` 파일을 만들거나 고치기 전에 `md_rules` skill 을 반드시 로드하고 그 규칙을 따른다.**

- 버전 표기, 제목 번호, 표와 figure 의 제목, code block 표기, 용어 규칙이 모두 그 skill 에 있다.
- 규칙 자체는 여기에 옮겨 적지 않는다. 두 곳에 적으면 한쪽만 바뀌어 어긋나므로, 형식이 달라지면 skill 만 고친다.
- 이 저장소는 그 규칙에 예외를 두지 않는다.
