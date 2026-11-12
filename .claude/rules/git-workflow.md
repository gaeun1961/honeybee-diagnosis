# Git 작업 순서

1. GitHub 이슈 먼저: 제목 `[기능] 짧은 설명`, 본문 2~3줄
2. 최신 main에서 브랜치: `{type}/{이슈번호}-{짧은설명}` (예: `chore/7-claude-setup`)
3. 커밋: 제목 `{type}: 설명 (#이슈번호)`, 본문은 아래 템플릿
4. PR: 본문은 커밋 메시지와 같은 템플릿, `Closes #이슈번호` 포함
5. 머지는 사용자가 하라고 할 때만. squash 말고 일반 merge commit. 머지 후 로컬 main pull

type: chore, docs, feature, fix, refactor, remove, test, UI

## 커밋·PR 템플릿
해당하는 type만 `[x]`.

```
## 어떤 기능인가요

## TODOS
- [ ]

## 참고할 만한 자료·기타사항

## type
- [ ] chore
- [ ] docs
- [ ] feature
- [ ] fix
- [ ] refactor
- [ ] remove
- [ ] test
- [ ] UI

Closes #

---

## 어려웠던 점(막힌점)

## 해결방법
```
