# Anomaly Detection Integration & CV Boilerplate Agent Instructions

이 저장소에서 AI 에이전트가 작업할 때 따르는 전체 지침이다.
요구사항의 배경과 근거는 `docs/dev/v0.2/BRIEF.md`와 `docs/guides/structure.md`에 둔다.

이 프로젝트는 범용 CV 실행 프레임워크 `cv_boilerplate` 위에 anomalib 기반 SOTA anomaly detection 모델(STFPM, EfficientAD)을 pure-PyTorch로 통합하고, 향후 Classification, Detection, Segmentation까지 확장 가능한 구조로 재설계한다.

---

## General Rules

- 이모지를 사용하지 않는다.
- 제품 코드는 Python으로 구현한다. 학습·평가·추론 로직은 pure-PyTorch로 작성하며 Lightning 등 상위 학습 프레임워크를 도입하지 않는다.
- 코드 내 주석은 영어로 작성한다. Markdown 문서는 한국어로 작성한다. 코드, 명령어, 파일 경로, 제품·라이브러리 고유 이름은 원문 표기를 유지한다.
- 사용자의 명시적인 요청 없이 코드나 문서를 생성하지 않는다.
- 대상 환경은 WSL2(Linux)이며 셸은 bash를 사용한다.
- Python 실행·검증은 conda 환경 `pytorch_env`에서 수행한다. 인터프리터 경로는 `/home/nampl/anaconda3/envs/pytorch_env/bin/python`이다.

  ```bash
  conda activate pytorch_env
  ```

---

## Code Style Rules

이 규칙은 이 프로젝트가 소유한 코드(`core/`, `tasks/`, `scripts/`, `configs/`)에 적용한다. `src/tasks/anomaly/models/`의 anomalib 원본 모델 코드에는 스타일 통일을 명목으로 수정하는 것을 엄격히 금지한다(원칙 1).

- 경로 표기는 `os.path` 방식을 사용하며 `pathlib.Path`를 사용하지 않는다.
- 네이밍은 PEP8을 따른다. 변수·함수는 `snake_case`, 클래스는 `PascalCase`, 상수는 `UPPER_SNAKE_CASE`를 사용한다.
- 멤버 변수에 접두사를 붙이지 않는다. private은 단일 언더스코어(`_`)만 허용한다.
- 등호와 콜론의 세로 정렬(vertical alignment)을 금지한다. dict 리터럴도 동일하다.
- 도메인 용어를 통일한다. split 명칭은 `valid`를 사용하고 `val`을 쓰지 않는다. 모델명과 백본명을 구분해 백본은 `backbone_name`으로 표기한다.

---

## Project Rules (3대 불변 원칙)

`BRIEF.md`의 세 원칙이 이 프로젝트의 최상위 규칙이다. 모델 코드는 불가침이고 boilerplate는 개선 대상이다.

### 원칙 1 — anomalib 모델 코드는 SSOT이며 수정하지 않는다

- anomaly detection 모델은 anomalib에 구현된 순수 PyTorch 코드를 **그대로 가져온다**.
- `src/tasks/anomaly/models/` 하위(예: `stfpm/`, `efficientad/`, `components/`)가 직접 불변의 SSOT(단일 진실 공급원) 역할을 수행한다. 별도의 `upstream/` 폴더는 두지 않는다.
- 해당 파일들에서 수정 가능한 부분은 **모듈 및 라이브러리 import 경로로 한정한다**. 네트워크 구조, 연산 순서, 하이퍼파라미터 기본값, 함수·클래스 시그니처, 반환 형식 변경은 모두 금지한다.
- 리팩터링, 코드 스타일 통일, 타입 힌트 추가, 사소한 정리를 포함해 원본을 손대는 모든 행위를 금지한다.
- 모든 모델은 `torch.nn.Module`을 직접 상속하며 `@MODELS.register` 팩토리 함수를 통해 프레임워크에 등록된다.
- **이 원칙은 사용자보다 AI 에이전트에게 우선 적용된다.** 통합 과정에서 문제가 생기면 모델 코드가 아니라 adapter 또는 boilerplate를 수정해 해결한다.

### 원칙 2 — Lightning은 절대 사용하지 않는다

- Lightning을 의존성으로 추가하지 않고, import하지 않으며, 어떤 실행 경로에서도 호출하지 않는다.
- `lightning_model.py`는 복사 대상이 아니라 **참고 대상**이다. optimizer·scheduler는 config 값으로, post-processing과 hook은 모델별 adapter로 옮겨 적는다.
- boilerplate는 고정된 것이 아니다. 성능 재현과 멀티태스크 확장을 위해 얼마든지 수정·개선될 수 있다.

### 원칙 3 — 데이터셋과 pretrained 가중치는 로컬 폴더의 것을 사용한다

- config가 가리키는 로컬 자산만 사용한다. 개발 머신의 기본값은 데이터셋 `/mnt/d/datasets`, pretrained 가중치 `/mnt/d/backbones`다.
- 경로는 config로 지정하며 코드에 하드코딩하지 않는다.
- **경로 자체는 머신마다 다를 수 있다.** config에는 `${paths.dataset_root}`·`${paths.backbone_root}` placeholder를 쓰고 실제 값은 `configs/local.yaml` 또는 환경변수 `DATASET_DIR`·`BACKBONE_DIR`로 받는다.
- 지정된 root가 없으면 개발 머신 기본값으로 조용히 폴백하지 않고 `ConfigError`로 즉시 실패한다.
- **에이전트는 데이터셋·모델·라이브러리를 자동으로 다운로드하거나 설치하지 않는다.** 필요한 자산이 없으면 무엇이 어느 경로에 필요한지 사용자에게 알리고, 준비될 때까지 해당 작업을 진행하지 않고 대기한다.

---

## 아키텍처 및 통합 구조 규칙

### 1. 프레임워크 코어 (`src/core/`)
- 공통 engine(`core/engine.py`)은 **Task-Agnostic**을 유지한다. 공통 루프에 모델명 또는 Task명으로 분기하는 조건문(`if task == 'anomaly'`)을 절대 두지 않는다.
- 모델별 차이는 wrapper가 아니라 **`TaskAdapter`의 lifecycle hook**(`train_step`, `eval_step`, `on_fit_start`, `on_fit_end`, `configure_optimizers` 등)으로 흡수한다.
- 실험 관리 도구(tensorboard, wandb 등)를 도입하지 않는다.

### 2. 태스크 패키지 구조 (`src/tasks/<task_name>/`)
모든 Task는 아래의 7개 컴포넌트 폴더로 완전 모듈화한다.
- `adapters/`: 베이스 어댑터(`base.py`) 및 모델별 전용 훅(`stfpm.py`, `efficientad.py`)
- `datasets/`: 데이터셋 베이스 및 구체 로더(`base.py`, `mvtec.py` 등)
- `models/<model_name>/`: 모델별 독립 패키지 (`torch.nn.Module` 직접 상속 및 `__init__.py` 등록)
- `transforms/`: 데이터 전처리/증강 빌더 (`default.py`)
- `metrics/`: 지표 계산기 (`image_auroc.py`, `pixel_auroc.py` 등)
- `losses/`: 복합 손실함수
- `postprocess/`: 시각화 및 후처리 (`visualizer.py`, `smoother.py`)

### 3. 설정 체계 (`configs/<task_name>/`)
- 각 Task 폴더 하위에 `data/`, `models/`, `batch/`, `_base.yaml`을 격리 배치한다.

### 4. 직접 실행형 CLI 진입점 (`scripts/`)
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`, `scripts/run_batch.py`를 단일 진입점으로 사용한다.
- **3단계 CLI 체계**:
  1. 1급 표준 인자: `--data`, `--model`, `--epochs`, `--batch-size`, `--output-dir`, `--checkpoint`, `--resume`, `--seed`, `--device` 등
  2. 동적 Selector: `--data.<key>`, `--model.<key>` (YAML 내 `selectors:` 템플릿과 연동)
  3. 범용 오버라이드: `--set <dotted.key>=<value>`

---

## 점진적 재구현 워크플로우 (Clean-Slate Incremental Rebuild Workflow)

v0.2는 안정 검증된 `main` 브랜치를 베이스라인으로 보존한 상태에서, `feat/v0.2-refactor` 브랜치에서 클린 상태(Clean Slate)로 시작하여 점진적으로 구축한다.

1. **선별적 파일 가져오기 및 작성**:
   - 필요한 기존 코드는 `git checkout main -- <path>`로 가져오거나, 새로운 아키텍처에 맞게 리팩터링하여 작성한다.
2. **최소 단위 스모크 검증**:
   - 구현된 최소 필수 파일 단위로 `python scripts/train.py ...` 소규모 스모크 실행을 수행하여 정상 동작을 확인한다.
3. **적대적 교차 검증 (Cross-vendor Review)**:
   - 주요 단위 구현 후 반대 벤더 CLI를 통한 적대적 검증을 거친다.
4. **단계별 보고 및 승인 커밋**:
   - 검증 결과를 사용자에게 보고하고 명시적 승인을 얻은 뒤 커밋을 진행한다.

---

## Document Rules

- 개발 문서는 `docs/dev/v{major}.{minor}/`에 둔다.
- 문서 체인은 `BRIEF.md`(사용자 의도와 원칙) → `PRD.md`(검증 가능한 요구사항) → `SPEC.md`(기술 결정) → `PLAN.md`(Phase 정의) → `backlog.json`(실행 단위) 순으로 파생된다.
- 사용자 요청으로 구현 또는 프로젝트 내용이 변경되면 `SPEC.md → PLAN.md → backlog.json → PRD.md` 순서로 갱신한다.
- `docs/refs/`는 이전 분석 결과를 보존하는 **읽기 전용 영역**이다. 필요한 부분만 참조하고 수정하지 않는다.
- 완료된 버전의 문서는 참조 전용으로 유지하며, 사용자의 명시적 요청 없이는 수정하지 않는다. 현재 진행 중인 버전이 v0.2이면 `docs/dev/v0.2/`만 갱신한다.
- Phase 완료 상태는 `backlog.json`의 각 Phase `status` 필드에서만 관리한다.

---

## Cross-vendor Adversarial Review Sub-agent

- 메인 에이전트는 Phase 구현을 끝낸 뒤, 마지막 실질 구현자와 반대 벤더의 검증만 담당하는 별도 서브 에이전트를 실행한다.
- 검토자는 파일을 수정하지 않는다.
- 각 지적은 `Critical / Major / Minor` 등급, 정확한 재현 조건, 위반한 SPEC 조항을 포함하며 심각도순으로 반환한다.
- 실행 횟수는 실패·오류·재검토를 모두 포함해 **최대 3회**로 제한한다.

### 이 프로젝트의 공격 초점
- **모델 SSOT 원본 훼손** — `src/tasks/anomaly/models/` 하위 파일이 import 경로 외에 원본과 다른가 (CON-001)
- **Lightning 잔재** — Lightning import, 의존성, 호출 경로가 남아 있는가 (CON-002)
- **anomalib 동등성 이탈** — adapter가 모델을 호출하는 방식이 계산 순서·데이터 흐름과 다른가
- **공통 engine 오염** — `core/`에 모델명·Task명 분기가 침투했는가 (NFR-005)
- **lifecycle hook 순서** — `on_fit_end`에서 모델 calibration이 올바르게 수행되는가
- **오프라인 위반** — 네트워크 다운로드를 유발하는 경로가 남아 있는가 (CON-003, CON-004)
- **학습/평가 누수** — train/valid/test 데이터가 누수되지 않는가
- **frozen 파라미터** — teacher가 항상 `eval()`로 유지되고 optimizer에서 제외되는가

---

## Commit and Push Rules

- 원격 저장소는 `https://github.com/nampluskr/defectvad-refactor`이다.
- 커밋 메시지에 작업 내용과 핵심 변경 사항을 명확히 포함한다.
- 커밋 전에는 관련 검증과 모델 SSOT 무결성 확인을 실행한다.
- 사용자의 명시적 승인 이후에만 커밋하고 푸시한다.
- 데이터셋, 백본 가중치, 체크포인트, 실험 산출물은 커밋하지 않는다.
- `docs/refs/`는 읽기 전용이며 변경을 커밋에 포함하지 않는다.
