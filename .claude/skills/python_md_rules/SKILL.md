---
name: python_md_rules
description: |-
  사용자가 python code에 대한 markdown document의 생성을 원할 때 로드 할 것.Python 소스(.py 파일, 패키지, 리포지토리)를 읽어 구조·공개 API·사용
    예시를 정리한 Markdown 문서(README, API 레퍼런스, 모듈 설명서)를
    생성한다. 사용자가 Python 코드를 두고 "문서 만들어줘", "README 써줘",
    "API 정리해줘", "모듈 설명서", "docstring 기반 문서화" 등을 요청하면
    반드시 이 스킬을 사용할 것. 'Markdown'이나 '문서'라는 단어를 쓰지
    않더라도, 코드에 대한 설명 산출물을 파일로 요구하면 사용한다.
    단, 코드 동작을 대화로 설명만 하는 경우, docstring·주석을 코드 안에
    추가하는 경우, Python 이외의 언어는 대상이 아니다.
---

# Documentation Conventions
Rev. 0 | Created: 2026-08-11 | Updated: 2026-08-11 12:45 CDT

## 1. Headings

+ H1 제목은 아래 순서로 만든다.
  1. Pipeline
  2. Method
  3. Input
  4. Output
  5. Result
  6. Analysis
+ Appendix 제목은 아래 순서로 만든다.
  1. Appendix A. Terminology
  2. Appendix B. CLI (Command Line Options)
+ 문서 머리에는 code goal과 background를 기술 할 것.
