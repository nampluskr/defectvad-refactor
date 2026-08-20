# Anomaly Detection Integration Agent Instructions

이 저장소에서 에이전트(Claude Code, Codex CLI)가 작업할 때 따르는 전체 지침이다.
요구사항의 배경과 근거는 `docs/dev/v0.1/BRIEF.md`에 둔다.

이 프로젝트는 범용 CV 실행 프레임워크 `cv_boilerplate` 위에 anomalib 기반 SOTA anomaly detection 모델을 통합한다. v0.1의 범위는 STFPM, EfficientAD, MVTec AD다.

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

## Code Style Rules

이 규칙은 이 프로젝트가 소유한 코드(`adapters/`, 공통 계층, config)에만 적용한다. `upstream/`에 복사한 anomalib 원본에는 **적용하지 않는다**. 스타일 통일을 이유로 원본을 손대는 것은 CON-001 위반이다.

- 경로 표기는 `os.path` 방식을 사용하며 `pathlib.Path`를 사용하지 않는다.
- 네이밍은 PEP8을 따른다. 변수·함수는 `snake_case`, 클래스는 `PascalCase`, 상수는 `UPPER_SNAKE_CASE`를 사용한다.
- 멤버 변수에 접두사를 붙이지 않는다. private은 단일 언더스코어(`_`)만 허용한다.
- 등호와 콜론의 세로 정렬(vertical alignment)을 금지한다. dict 리터럴도 동일하다.
- 도메인 용어를 통일한다. split 명칭은 `valid`를 사용하고 `val`을 쓰지 않는다. 모델명과 백본명을 구분해 백본은 `backbone_name`으로 표기한다.

## Project Rules

`BRIEF.md`의 세 원칙이 이 프로젝트의 최상위 규칙이다. 원칙 1과 2는 비대칭이며, 모델 코드는 불가침이고 boilerplate는 개선 대상이다.

### 원칙 1 — anomalib 모델 코드는 SSOT이며 수정하지 않는다

- anomaly detection 모델은 anomalib에 구현된 순수 PyTorch 코드를 **그대로 가져온다**. `torch_model.py`, `loss.py`, `anomaly_map.py` 및 함께 복사하는 components가 모두 해당한다.
- `upstream/` 하위 파일에서 수정 가능한 부분은 **모듈 및 라이브러리 import 경로로 한정한다**. 네트워크 구조, 연산, 하이퍼파라미터 기본값, 함수·클래스 시그니처, 반환 형식 변경은 모두 금지한다.
- 리팩터링, 코드 스타일 통일, 타입 힌트 추가, 사소한 정리를 포함해 원본을 손대는 모든 행위를 금지한다.
- **이 원칙은 사용자보다 AI 에이전트에게 우선 적용된다.** 통합 과정에서 문제가 생기면 모델 코드가 아니라 adapter 또는 boilerplate를 수정해 해결한다. 성능이 재현되지 않아도 마찬가지다.

### 원칙 2 — Lightning은 절대 사용하지 않는다

- Lightning을 의존성으로 추가하지 않고, import하지 않으며, 어떤 실행 경로에서도 호출하지 않는다.
- `lightning_model.py`는 복사 대상이 아니라 **참고 대상**이다. optimizer·scheduler는 config 값으로, post-processing과 hook은 모델별 adapter로 옮겨 적는다.
- boilerplate는 고정된 것이 아니다. 성능 재현을 위해 얼마든지 수정·개선될 수 있다.

### 원칙 3 — 데이터셋과 pretrained 가중치는 로컬 폴더의 것을 사용한다

- config가 가리키는 로컬 자산만 사용한다. 개발 머신의 기본값은 데이터셋 `/mnt/d/datasets`, pretrained 가중치 `/mnt/d/backbones`다.
- 경로는 config로 지정하며 코드에 하드코딩하지 않는다.
- **경로 자체는 머신마다 다를 수 있다.** 이 저장소는 압축 배포로 여러 로컬 환경에서 사용되므로, config에는 `${paths.dataset_root}`·`${paths.backbone_root}` placeholder를 쓰고 실제 값은 `configs/local.yaml` 또는 환경변수 `DATASET_DIR`·`BACKBONE_DIR`로 받는다(SPEC §6). placeholder를 절대 경로로 되돌리지 않는다.
- 지정된 root가 없으면 개발 머신 기본값으로 조용히 폴백하지 않고 `ConfigError`로 즉시 실패한다.
- **에이전트는 데이터셋·모델·라이브러리를 자동으로 다운로드하거나 설치하지 않는다.** 필요한 자산이 없으면 무엇이 어느 경로에 필요한지 사용자에게 알리고, 준비될 때까지 해당 작업을 진행하지 않고 대기한다.
- anomalib 원본에 포함된 다운로드 로직을 코드 수정으로 제거하지 않는다(원칙 1). 로컬 자산을 지정해 그 경로를 타지 않도록 adapter 또는 boilerplate에서 처리한다.

### 통합 구조 규칙

- 공통 engine(`core/engine.py`)은 task-agnostic을 유지한다. 공통 루프에 모델명 또는 task명으로 분기하는 조건문을 두지 않는다.
- 모델별 차이는 wrapper가 아니라 **adapter가 흡수한다**. 모델 자리에는 anomalib `torch_model.py`의 `nn.Module`이 그대로 들어간다.
- 모델별 lifecycle(auxiliary 데이터, 통계 계산, calibration)은 `TaskAdapter`의 hook으로 흡수한다. 공통 engine의 시그니처를 바꾸지 않는다.
- 두 모델 이상에서 같은 lifecycle 필요성이 확인되기 전에는 새 추상화를 추가하지 않는다.
- 실험 관리 도구(tensorboard, wandb 등)를 도입하지 않는다.
- 새 의존성을 추가하기 전 기존 스택(torch, torchvision, torchmetrics)으로 구현 가능한지 확인하고 사용자 승인을 받는다.
- `BRIEF.md`에 확정된 범위, 대상 모델, 데이터셋을 임의로 변경하지 않고 불필요한 추상화 계층을 추가하지 않는다.

## 성능 판정 규칙

- **에이전트는 학습·평가를 임의로 장시간 실행하지 않는다. 실행 주체는 사용자다.**
- 에이전트는 구현을 마친 뒤 사용자가 터미널에서 실행할 명령어를 제시하고, 사용자가 피드백한 결과를 anomalib reference와 비교한다.
- 성능 tolerance를 문서에서 미리 고정하지 않는다. 사용자가 재현 성공으로 판단하면 해당 Phase를 완료한다.
- 차이가 발생하면 모델 구현 차이로 단정하지 않고 `PRD.md` §5.4의 항목(preprocessing, augmentation, optimizer/scheduler, epoch/batch, pretrained weight, score normalization, post-processing, threshold, metric, evaluation protocol)을 먼저 확인한다.
- 원인이 무엇이든 모델 코드는 수정하지 않는다(CON-001). adapter 또는 boilerplate에서 해결한다.
- 동작 확인 목적의 스모크 실행은 단일 카테고리 소규모로 수행한다. 전체 15개 카테고리 벤치마크는 v0.1의 완료 조건이 아니다.

## Document Rules

- 개발 문서는 `docs/dev/v{major}.{minor}/`에 둔다.
- 문서 체인은 `BRIEF.md`(사용자 의도와 원칙) → `PRD.md`(검증 가능한 요구사항) → `SPEC.md`(기술 결정) → `PLAN.md`(Phase 정의) → `backlog.json`(실행 단위) 순으로 파생된다.
- 사용자 요청으로 구현 또는 프로젝트 내용이 변경되면 `SPEC.md → PLAN.md → backlog.json → PRD.md` 순서로 갱신한다.
- `docs/refs/`는 이전 분석 결과(BRIEF, PRD, SPEC, PLAN, WIKI_INDEX, backlog.json, comparison/)를 보존하는 **읽기 전용 영역**이다. 필요한 부분만 참조하고 수정하지 않는다.
- 세 저장소 비교 분석은 `docs/refs/comparison/`에 이미 작성되어 있다. 새로 분석하지 말고 `PLAN.md` §1.1의 색인에서 필요한 문서의 해당 절만 조회한다.
- 완료된 버전의 문서는 참조 전용으로 유지하며, 사용자의 명시적 요청 없이는 수정하지 않는다. 현재 진행 중인 버전이 v0.2이면 `docs/dev/v0.2/`만 갱신하고 `docs/dev/v0.1/`은 형식과 과거 결정의 참고 목적으로만 읽는다.
- Phase 완료 상태는 `backlog.json`의 각 Phase `status` 필드에서만 관리한다. `README.md`, `PLAN.md`, `SPEC.md`, `PRD.md`에는 Phase 상태를 기록하지 않는다.
- 문서(`BRIEF.md`, `PRD.md`, `SPEC.md`, `PLAN.md`)도 사용자 요청이 있으면 반대 벤더 CLI로 적대적 검증을 받는다. 검토자 선정 기준은 Phase 검증과 같고, 검토자는 대상 문서와 상위 문서, `AGENTS.md`만 읽는다.
- 문서 검증 역시 실행과 보완을 합쳐 최대 3회로 제한한다. 상세는 아래 Verification Attempt Limit을 따른다.

### backlog.json 구조

`backlog.json`은 최상위에 프로젝트 메타와 정책을, `phases` 배열에 실행 단위를 담는다.

| 키 | 내용 |
|---|---|
| `schema_version`, `version`, `created`, `scope` | 스키마 버전과 프로젝트 메타 |
| `remote_repository` | 원격 저장소 URL |
| `source_documents` | 파생 원본 문서 경로 |
| `status_values` | 허용 상태 값 |
| `global_constraints` | 전 task에 적용되는 제약(CON-001~005, NFR-005, PRD §5) |
| `phases[]` | `id`, `title`, `status`, `depends_on`, `exit_criteria`, `tasks` |
| `phases[].tasks[]` | `id`, `title`, `status`, `description`, `prd_refs`, `spec_refs`, `depends_on`, `parallelizable`, `parallel_group`, `execution_mode`, `suggested_agent_role`, `scope.allowed`, `scope.avoid`, `verification`, `completion_criteria`, `artifacts` |

각 task의 `scope.avoid`에는 해당 task에서 특히 위험한 제약을 반복 기재한다. `spec_refs`의 절 번호(예: `SPEC §4.5`)가 적대적 검증이 참조하는 조항 ID다.

## Common Contract and Agent Execution Rules

- P0(기준 확정), P1(공통 구조), P6(구조 정리)은 메인 세션이 직접 수행한다. 모델 통합 Phase(P2, P4)는 모델별 에이전트에 위임할 수 있다.
- 모델 에이전트는 자기 모델의 `adapters/<model>.py`와 해당 config 외의 파일을 수정하지 않는다.
- 공통 코드(`core/`, `tasks/anomaly/adapter.py`, `cli/`, `bench/`) 수정 권한은 메인 세션만 갖는다. 하위 에이전트는 직접 수정하지 않고 문제, 재현 조건, 최소 수정안, 영향받는 SPEC 조항 번호를 담은 변경 요청을 반환한다.
- `upstream/` 하위는 **어떤 에이전트도 수정하지 않는다.** 복사(P1-T01) 시점의 import 경로 조정이 유일한 예외다.
- 공통 코드 변경은 등급 A(계약 무변경) / B(하위호환 확장) / C(계약 변경)로 구분한다. C는 사용자 승인 후에만 진행한다.
- 어떤 등급이든 공통 루프에 모델명·task명으로 분기하는 수정은 허용하지 않는다. adapter 또는 hook으로 흡수한다.
- 공통 코드가 바뀌면 그 시점까지 완료된 모든 모델의 스모크를 재실행해 회귀를 확인한다. `tasks/anomaly/` 외 4개 task(classification, detection, segmentation, toy)가 같은 공통 계층을 공유하므로, `core/` 변경 시 이들의 영향도 확인한다.
- Phase는 `backlog.json`의 `depends_on`을 지켜 순차 진행한다. 같은 `parallel_group`의 task만 동시에 진행한다.

## Phase Execution Workflow

각 Phase는 다음 순서로 완료한다.

1. 해당 Phase의 task `scope`, `verification`, `completion_criteria`를 구현하고 검증한다. 검증에는 소규모 스모크 실행을 포함한다. 성능 판정이 필요한 Phase(P3, P5)는 사용자에게 실행 명령어를 제시하고 결과를 받는다.
2. 마지막 실질 구현자의 **반대 벤더 CLI**에 적대적 검증을 위임한다. Codex 구현은 Claude Sonnet headless CLI가, Claude Code 구현은 Codex CLI가 검토한다. 토큰 한도로 구현자가 Phase 중간에 바뀌면 마지막 실질 구현자를 기준으로 검토자를 다시 정한다.
3. 유효한 Critical 지적은 모두 수정하고 관련 검증을 재실행한다. Critical을 수정했다면 같은 반대 벤더 검토를 한 번 더 실행해 해소를 확인한다. Major와 Minor는 처리 여부와 근거를 기록한다. 실행 횟수는 아래 Verification Attempt Limit을 따른다.
4. `docs/dev/v{major}.{minor}/reviews/A{n}.md`에 구현자, 검토 모델, 대상 파일, 실행 일시, 심각도별 건수, 지적·재현 조건·관련 SPEC 조항·처리 상태를 기록한다. 유효하지 않은 지적의 반박 근거도 기록한다.
5. 변경 내용, 검증 결과, 교차 검토 결과와 보완 조치, 남은 위험을 사용자에게 보고하고 커밋·푸시 승인을 요청한다. 검토 결과는 지적사항별 심각도, 근거, 처리 상태를 포함한 Markdown 표로 제시한다.
6. 사용자의 명시적 승인을 받은 후에만 해당 Phase 변경을 커밋한다.

적대적 검증 필수 통과 Phase는 P1(공통 구조), P2·P4(모델 통합), P5(완료 조건 전수 검증)다. 필수 Phase에 미해결 Critical이 있으면 다음 Phase로 진행하지 않는다.

필요한 반대 벤더 CLI를 사용할 수 없는 환경이면 그 사실과 사유를 사용자에게 알리고, 대체 검증 방안을 제시한 뒤 승인을 요청한다. 교차 검증을 생략하거나 사용자 승인 전에 커밋하지 않는다.

## Verification Attempt Limit

적대적 검증 실행 횟수는 Phase 검증과 문서 검증에 동일하게 적용한다.

- 하나의 검증 대상(Phase 1개 또는 문서 1개)에 대한 검토 CLI 실행은 **실패·오류·프롬프트 재작성·재검토를 모두 포함해 최대 3회**로 제한한다.
- 지적 보완 후의 재검증도 이 3회에 포함된다. 보완 사이클을 별도로 세지 않는다.
- 3회를 소진하면 추가 실행 대신 마지막 유효 검토 결과, 반영한 보완 내용, 미해결 지적과 남은 위험을 기록하고 사용자에게 보고한다.
- 3회 소진 시점에 미해결 Critical이 남아 있으면 다음 Phase로 진행하지 않고 사용자 판단을 요청한다.
- 실행 횟수와 각 회차의 결과(모델, 실행 일시, 유효 여부, 사유)를 검토 기록 문서에 남긴다.

## Cross-vendor Adversarial Review Sub-agent

- 메인 에이전트는 Phase 구현을 끝낸 뒤, 마지막 실질 구현자와 반대 벤더의 검증만 담당하는 별도 서브 에이전트를 실행한다.
- 서브 에이전트는 현재 Phase, 지정 제품 소스 파일, 해당 Phase의 공격 초점, `spec_refs`만 프롬프트에 포함한다. 구현 세션 대화나 판단 근거는 전달하지 않는다.
- 검토자는 파일을 수정하지 않는다. 문서, 설정, 실험 산출물, Git 상태·브랜치·원격 저장소·커밋 이력과 셸 도구를 요청하거나 사용하지 않는다.
- 각 지적은 `Critical / Major / Minor` 등급, 정확한 재현 조건, 위반한 SPEC 조항을 포함하며 심각도순으로 반환한다.
- 검토자의 지적은 메인 에이전트가 검토한다. 유효하지 않은 지적은 근거와 함께 기록한다.
- 필요한 CLI의 응답 지연, 인증 실패, 네트워크 오류 등으로 검증하지 못하면 서브 에이전트는 오류 내용과 대체 검증안을 메인 에이전트에 반환한다.
- 문서 검증 시에는 검토자가 대상 문서와 상위 문서, `AGENTS.md`를 읽기 위해 읽기 전용 셸 명령(`cat`, `sed -n` 등)을 사용하는 것을 허용한다. Codex CLI에는 별도 파일 읽기 도구가 없어 셸을 전면 금지하면 검토가 불가능하다.
- 검토 CLI를 실행하는 외부 명령의 시간 제한은 기본 10분으로 설정한다.

### 이 프로젝트의 공격 초점

이 프로젝트의 적대적 검증은 다음 축을 우선 공격 대상으로 삼는다. 앞의 세 축이 이 프로젝트 고유의 것이며 가장 중요하다.

- **upstream 원본 훼손** — `upstream/` 하위 파일이 import 경로 외에 원본과 다른가. 네트워크 구조, 연산 순서, 하이퍼파라미터 기본값, 시그니처, 반환 형식이 바뀌었는가. 스타일 통일이나 타입 힌트 추가 명목의 변경이 섞여 있는가 (CON-001)
- **Lightning 잔재** — Lightning import, 의존성, 호출 경로가 남아 있는가. `lightning_model.py`의 내용이 코드로 복사되지 않고 config와 adapter hook으로 올바르게 이관되었는가 (CON-002)
- **anomalib 동등성 이탈** — adapter가 upstream을 호출하는 방식이 `lightning_model.py`의 계산 순서·데이터 흐름과 다른가. loss 조합, feature 정규화, anomaly map 합성, score 산출이 원본과 어긋나는가
- **공통 engine 오염** — `core/` 또는 공통 `AnomalyAdapter`에 모델명·task명 분기가 침투했는가. EfficientAD 전용 개념(auxiliary loader, 분위수)이 공통 계층으로 새어 나왔는가 (NFR-005)
- **lifecycle hook 순서** — `on_fit_end`에서 모델 calibration이 threshold 계산보다 먼저 끝나는가. `super().on_fit_end()` 호출이 누락되지 않았는가. calibration이 best 가중치 기준으로 수행되는가 (SPEC §4.3)
- **오프라인 위반** — 네트워크 다운로드를 유발하는 경로가 남아 있는가. `weights=` 인자, `torch.hub`, 자동 다운로드 로직이 실행 경로에 있는가. 로컬 자산 부재 시 조용히 랜덤 초기화로 폴백하지 않는가 (CON-003, CON-004)
- **학습/평가 누수** — teacher 통계는 train만, 분위수와 threshold는 valid만 사용하는가. test가 어느 단계에도 개입하지 않는가. 평가 시 `model.eval()`·`torch.no_grad()`와 metric 리셋이 누락되지 않았는가
- **frozen 파라미터** — teacher가 항상 `eval()`로 유지되어 BatchNorm running stats가 갱신되지 않는가. teacher 파라미터가 optimizer에 포함되지 않는가
- **재현성** — seed 고정 범위, config·metric 기록이 실제 재현을 보장하는가. 동일 config·seed로 결과가 재현되는가 (AC-009)

### 검토 명령

`<phase>`, `<changed-files>`, `<adversarial-focus>`, `<spec-refs>`는 현재 작업 내용으로 대체한다.

Claude Code가 마지막 구현자일 때 Codex 검토:

```bash
codex exec --model gpt-5.6-sol --sandbox read-only --cd "/mnt/d/projects/nampluskr/00_review/260820_defectvad-refactor" "You are an adversarial reviewer for <phase>. Your job is to break this code, not to confirm it works. Review only these product source-code files: <changed-files>. Attack these specific points: <adversarial-focus>. Validate against these spec clauses: <spec-refs>. Do not inspect documentation, configuration, experiment artifacts, Git status, branches, remotes, or commit history. For each finding, report severity (Critical/Major/Minor), exact reproduction conditions, and the violated spec clause. Order findings by severity. Do not modify files."
```

Codex가 마지막 구현자일 때 Claude Sonnet 검토:

```bash
claude -p "You are an adversarial reviewer for <phase>. Your job is to break this code, not to confirm it works. Review only these product source-code files: <changed-files>. Attack these specific points: <adversarial-focus>. Validate against these spec clauses: <spec-refs>. Do not inspect documentation, configuration, experiment artifacts, Git status, branches, remotes, commit history, or use Bash or any shell tool. For each finding, report severity (Critical/Major/Minor), exact reproduction conditions, and the violated spec clause. Order findings by severity. Do not modify files." --model sonnet --safe-mode --allowedTools "Read,Glob,Grep" --disallowedTools "Edit,Write,Bash" --permission-mode dontAsk --max-turns 5 --output-format json --no-session-persistence
```

Claude 검토는 `Read`, `Glob`, `Grep`만 허용하며 파일 변경과 셸 도구를 금지한다. 모델 접근이 거부되면 기본 모델로 조용히 폴백하지 않고 오류와 대체안을 보고한다. 필요하면 `--max-budget-usd`로 호출별 비용 상한을 둔다.

### upstream 무결성 검증

`upstream/` 하위는 적대적 검증과 별개로 매 Phase 완료 시 기계적으로 확인한다. 검토 CLI의 판단에 의존하지 않는다.

- P1-T01에서 복사 시점의 원본과 diff해 import 문 외 차이가 없음을 확인하고 결과를 기록한다.
- 이후 Phase에서는 P1-T01 시점의 `upstream/` 트리와 현재 트리를 비교해 변경이 없음을 확인한다.
- 변경이 발견되면 즉시 되돌린다(CON-001). 변경이 필요해 보이는 상황은 adapter 또는 boilerplate 수정으로 해결한다.

## Commit and Push Rules

- 원격 저장소는 `https://github.com/nampluskr/defectvad-refactor`로 확정되었다(2026-08-20 사용자 확인). `backlog.json`의 `remote_repository`에 기록한다.
- 로컬 저장소의 원격 연결(`git remote add`)과 첫 푸시는 사용자 승인 아래 수행한다. 그 전에는 커밋하지 않는다.
- 커밋은 하나의 완료된 Phase에 대응하며, 커밋 메시지에 Phase 번호와 핵심 변경 사항을 포함한다.
- 커밋 전에는 해당 Phase의 관련 검증과 `upstream/` 무결성 확인을 실행한다.
- Phase 완료, 반대 벤더 교차 검증, 지적사항 수정 및 재검증을 마친 뒤 사용자에게 커밋 및 푸시 승인을 요청한다.
- 사용자의 명시적 승인 이후에만 커밋하고 푸시한다.
- 다른 작업의 변경 사항을 임의로 포함, 되돌리기, 삭제하지 않는다.
- 데이터셋, 백본 가중치, 체크포인트, 실험 산출물은 커밋하지 않는다.
- `docs/refs/`는 읽기 전용이다. 이 폴더의 변경을 커밋에 포함하지 않는다.
