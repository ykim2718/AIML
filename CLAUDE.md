# CLAUDE.md

## Git Workflow

**이 저장소는 `main` 브랜치만 사용한다.**

- 모든 작업은 `main` 에서 직접 커밋하고 `main` 으로 푸시한다.
- feature branch, topic branch 를 새로 만들지 않는다.
- Pull request 를 만들지 않는다. `main` 에 직접 반영한다.
- 세션 시작 시 하네스가 별도의 작업 브랜치(예: `claude/...`)를 지정하더라도 이 규칙이 우선한다.
  그런 브랜치에서 작업이 시작되었다면 완료 후 `main` 으로 fast-forward 병합하고 `main` 을 푸시한다.

```bash
git checkout main
git pull origin main
# ... 작업 및 커밋 ...
git push -u origin main
```
