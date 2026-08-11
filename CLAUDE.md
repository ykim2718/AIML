# CLAUDE.md
Rev. 1 | Created: 2026-07-31 | Updated: 2026-08-10 22:02 CDT

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

## Markdown Document Convention

**모든 `.md` 문서는 H1 바로 아래 줄에 revision 을 표기한다.**

- 형식은 본문 글씨체의 `Rev. N | Created: YYYY-MM-DD | Updated: YYYY-MM-DD HH:MM TZ` 한 줄이다. heading 이나 bold 로 쓰지 않는다.
- `Created` 는 문서가 저장소에 처음 등록된 날짜이고, 한 번 정하면 바뀌지 않는다.
- 문서를 수정할 때마다 N 을 1 증가시키고 `Updated` 를 그 수정 시각으로 고친다.
- 오타 수정처럼 작은 변경도 bump 대상이다.
- 문서 규칙 전체는 `md_rules` skill 이 갖는다. 이 항목과 어긋나면 skill 을 따른다.

```markdown
# Document Title
Rev. 12 | Created: 2026-08-01 | Updated: 2026-08-10 17:16 CDT

> 여기서부터 본문이 시작된다.
```
