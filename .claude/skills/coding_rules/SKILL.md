---
name: coding_rules
description: 코드를 쓰거나 고칠 때 반드시 지킬 규칙. Edit/Write 하기 전에 로드할 것. 입력 검증, 실패 처리, API 설계, 검증, 버전 표기에 적용된다.
---

# 코딩 규칙

모든 규칙의 뿌리는 하나다: **모순이나 이상을 만나면 조용히 넘어가지 말고 드러내라.**
조용히 넘어간 것은 반드시 나중에, 더 나쁜 모습으로 돌아온다.

## 1. 원칙

### 1.1. 애매하면 조용히 고르지 말고 에러를 내라

호출자가 모순된 입력을 줬다면 그건 버그다. 한쪽을 골라주는 건 버그를 감추는 것이다.

- 같은 것을 정하는 인자 2개를 동시에 받으면 → 에러. 한쪽 무시 금지.
- 예상 밖 입력을 기본값으로 때우지 마라.
- 침묵의 우선순위(`a or b`)로 모순을 해소하지 마라.

```python
# 나쁨: 둘 다 줬는데 value가 이긴다. 호출자는 method가 먹은 줄 안다
result = value or compute(method=method)

# 좋음
if value and method:
    raise ValueError("value and method are mutually exclusive; pass one or neither.")
```

### 1.2. 실패를 삼키지 마라

로그가 없는 실패는 없었던 일이 된다. 몇 달을 그렇게 지나갈 수 있다.

- `verbose`/`debug` 플래그 안에 에러 로그를 넣지 마라. 실패는 항상 보이게.
- 빈 결과를 정상처럼 반환하지 마라. 0건은 그 자체가 신호다.
- 참/거짓만 보고 통과시키지 마라. 공백 문자열도 truthy다.
- `except: pass` 금지. 삼킬 거면 왜 삼켜도 되는지 주석으로 남겨라.

```python
# 나쁨: verbose=False면 아무도 모른다
if verbose:
    logger.warning(f"failed: {reason}")

# 좋음: 실패는 verbose와 무관하게 보인다
logger.error(f"failed {failed_count}/{total}: {reason}")
```

```python
# 나쁨: '\r\n' * 6 도 truthy라 통과한다
if response.text:
    data = response.json()

# 좋음
if not response.text.strip():
    raise ValueError(f"empty body: {response.status_code=}")
```

### 1.3. 잘못된 사용이 애초에 불가능하게 만들어라

막는 것보다 없애는 게 낫다. 에러로 막으면 여전히 틀릴 수 있지만, 표현 자체가 불가능하면 틀릴 수 없다.

- 인자 2개가 배타적이면 → **하나로 합쳐라.** 그게 안 될 때만 에러로 막아라.
- "쓰면 안 되는 조합"을 문서에 적지 말고 코드가 막게 하라.
- 위험한 기본값을 주지 마라. 위험한 선택은 명시적으로 opt-in 하게 하라.

### 1.4. 검증 없이 "됐습니다" 하지 마라

- 고쳤으면 실제로 돌려봐라. 못 돌렸으면 **"안 돌려봤다"고 말해라.**
- 테스트가 실패하면 기대값이 틀린 건지 코드가 틀린 건지 **먼저 가려라.**
- 기존 동작을 바꿨으면 기존 사용처가 안 깨졌는지부터 확인하라.

### 1.5. import는 파일 맨 위에. 함수 안에 넣지 마라

의존성은 파일만 열면 보여야 한다. 함수 안에 숨긴 import는 그 함수가 호출되는 순간에야
ModuleNotFoundError로 터진다 — 프로세스가 한참 돌아간 뒤에.

- 무거운 의존성을 피하고 싶으면 import를 숨기지 말고 **모듈을 분리하라.**

### 1.6. 남의 코드를 고칠 땐 계약을 지켜라

- 반환 타입/시그니처를 바꾸면 호출자를 전부 찾아 고쳐라.
- 기존 데이터·저장 포맷과의 호환을 깨는 변경은 먼저 알리고 승인받아라.
- 기능 추가는 기존 경로를 건드리지 않는 방향으로 설계하라.

## 2. 공통

### 2.1. Author

- author는 **yRocket**을 사용할 것.

### 2.2. Versioning

- versioning marker는 날짜가 있는 `__version__` `Major.Minor.Patch.Date(YYYY.M.D)` 형식이
  default이고, 날짜가 없는 `Major.Minor.Patch` 형식도 있다.
- `py`, `ps1`, `sh`, `yml` 파일 머리에 아래처럼 versioning marker를 기입하고, 없으면 추가할 것:

  ```python
  __version__ = "0.0.0.2026.7.14"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)
  ```

- 날짜가 있는 형식이면 change 발생 시:
  - **patch bump** + 날짜를 오늘로
  - 기능 추가면 **minor bump**, patch는 0
  - docstring changelog에 한 줄 추가
- 날짜가 없는 형식 (예: `__version__ = "0.0.0"`) 이면 change 발생 시 **patch bump**만 할 것
  (날짜 없음).

## 3. Python 작성

### 3.1. Standard Module

- 다음 모듈을 최대한 사용할 것: pathlib, argparse, tqdm, typing

### 3.2. Annotation

- `from typing import Union` 등을 사용해서 annotation을 표시할 것.
- function annotation에서 `array: list = None`의 형식을 사용할 것.

### 3.3. Style

- `from tqdm import tqdm`을 아래처럼 사용할 것.

  ```python
  pbar = tqdm(files, ncols=100, unit='parquet')
  for fpath in pbar:
      pbar.set_description(f"Reading {pathlib.Path(fpath).stem}")
  ```

### 3.4. Overall Rule

- hard coding 하지 말 것.
- main() 함수를 만들지 말 것.
- max code + comment line width <= 120
- python interpreter (64-bit 기본): `c:\Y\anaconda3\python.exe`
- python interpreter (32-bit, daishin/Creon 전용): `C:\Y\anaconda3\envs\python32\python.exe`
- comment를 모두 영어로 쓸 것.

### 3.5. Naming

- dir을 쓰지 말고 folder를 사용할 것.

### 3.6. CLI

- CLI option parsing을 `def parse_args() -> argparse.Namespace`에서 할 것.
- `parse_args()`는 `if __name__ == '__main__':` 바로 위에 위치시킬 것.
- CLI option이 하나도 없을 때는 CLI 사용법을 출력할 것.
- `argparse.ArgumentParser`의 choices 옵션을 필요시 사용할 것.
- CLI option name에는 white space를 채울 때 dash(`-`)를 사용할 것.
- **Use hyphens instead of underscores for all CLI options.**
  `-plain_text`가 아니라 `-plain-text`다. 발견하면 모두 고칠 것 (python parameter name은 그대로다).
- CLI option value들 사이의 조건 검증은 `parse_args()`에서 진행하고 crash 처리할 것.
- CLI option에서 choices로 검증된 변수를 함수의 argument로 사용할 때는
  `from typing import Literal`을 사용해서 표시할 것.
- file or folder path의 CLI는 `parse_args()`에서 확인하고 오류가 있으면 help 메세지 출력할 것.
- `import click` 사용 시, 복수의 option이 배타적일 때는 `MutuallyExclusiveOption(click.Option)`을
  `cls`로 걸어서 parsing 단계에서 막을 것.

### 3.7. Reproducibility

- `list(set(...))`를 사용하지 말고 `sorted(set(...))`을 쓸 것.

### 3.8. Data

- pandas DataFrame을 리턴하는 함수는 docstring에 리턴하는 pd.DataFrame의 index와 columns의
  name을 명기할 것.

### 3.9. Function

- 함수 정의할 때 될 수 있으면 keyword arguments를 사용하여 readability를 높일 것.
- 함수 호출할 때 keyword arguments 방식을 사용할 것.

### 3.10. 환경

- interpreter 경로는 3.4 Overall Rule 참조.
- 스크립트 단독 실행 시 `PYTHONPATH`에 repo 루트가 필요하다.

### 3.11. core 폴더는 공용이다

`y/core/`는 모든 프로젝트가 쓰는 공용 코드다. 특정 프로젝트를 상관하는 코멘트를 넣지 마라.
공용 코드가 소비자를 알면, 소비자가 바뀔 때마다 그 코멘트는 거짓말이 된다.

```python
# 나쁨: core 파일이 특정 프로젝트를 앎
import docker  # the REST API image installs it via requirements.txt

# 좋음
import docker  # pip install docker
```

## 4. 반면교사: 실제 사고 기록

전부 실제로 낸 사고에서 나왔다. 원인이 셋 다 같다 — **모순을 만나면 에러 대신 조용히 한쪽을 고름.**

| 사고 | 결과 |
|---|---|
| 빈 응답을 truthy라 통과시킴 | 스케줄러가 죽음 |
| `verbose=False`라 실패를 로그도 안 남김 | 수집이 7개월간 조용히 멈춰 있었음 |
| 배타적 인자 2개를 받고 한쪽을 조용히 무시 | 사용자가 안 먹은 옵션을 먹은 줄 앎 |
