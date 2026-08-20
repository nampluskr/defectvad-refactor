# Anomaly Detection Integration Agent Instructions

Claude Code가 이 저장소에서 작업할 때 매 턴 지켜야 하는 규칙이다. 배경과 근거는 `docs/dev/v0.1/` 문서에 둔다. 전체 지침은 `AGENTS.md`를 참조한다.

이 프로젝트는 `cv_boilerplate` 위에 anomalib 기반 anomaly detection 모델을 통합한다. v0.1 범위는 STFPM, EfficientAD, MVTec AD다.

## 최상위 원칙 세 가지

1. **anomalib 모델 코드는 SSOT이며 수정하지 않는다.** `upstream/` 하위에서 허용되는 변경은 import 경로뿐이다. 리팩터링, 스타일 통일, 타입 힌트 추가를 포함해 그 외 모든 변경을 금지한다. **이 원칙은 사용자보다 AI 에이전트에게 우선 적용된다.** 문제가 생기면 모델이 아니라 adapter 또는 boilerplate를 고친다.
2. **Lightning은 절대 사용하지 않는다.** 의존성 추가·import·호출을 모두 금지한다. `lightning_model.py`는 복사 대상이 아니라 참고 대상이며, optimizer·scheduler는 config로 hook은 adapter로 옮겨 적는다.
3. **config가 가리키는 로컬 자산만 쓴다.** 개발 머신 기본값은 `/mnt/d/datasets`, `/mnt/d/backbones`다. 경로는 머신마다 다를 수 있으므로 config에는 `${paths.*}` placeholder를 쓰고 실제 값은 `configs/local.yaml` 또는 환경변수로 받는다(SPEC §6). **에이전트는 자동 다운로드·설치를 하지 않는다.** 없으면 무엇이 어느 경로에 필요한지 알리고 대기한다.

## General Rules

- 이모지를 사용하지 않는다.
- 제품 코드는 Python으로 구현하고 학습·평가·추론 로직은 pure-PyTorch로 작성한다.
- 코드 내 주석은 영어로, Markdown 문서는 한국어로 작성한다. 코드·명령어·경로·고유 이름은 원문 표기를 유지한다.
- 사용자의 명시적인 요청 없이 코드나 문서를 생성하지 않는다.
- 대상 환경은 WSL2(Linux)이며 셸은 bash를 사용한다. Python 실행·검증 전에 conda 환경 `pytorch_env`를 활성화한다.
- 경로 표기는 `os.path` 방식을 사용하며 `pathlib.Path`를 사용하지 않는다.
- PEP8 네이밍을 따르고 멤버 변수 접두사와 등호·콜론 세로 정렬을 금지한다. split은 `valid`, 백본은 `backbone_name`으로 표기한다.
- **코드 스타일 규칙은 `upstream/`에 적용하지 않는다.** 스타일을 이유로 원본을 손대는 것은 원칙 1 위반이다.

## Project Rules

- 공통 engine(`core/engine.py`)은 task-agnostic을 유지한다. 공통 루프에 모델명·task명 분기를 두지 않는다.
- 모델별 차이는 wrapper가 아니라 adapter가 흡수한다. 모델 자리에는 anomalib `torch_model.py`의 `nn.Module`이 그대로 들어간다.
- 모델별 lifecycle(auxiliary 데이터, 통계, calibration)은 `TaskAdapter` hook으로 흡수하고 공통 engine 시그니처를 바꾸지 않는다.
- 두 모델 이상에서 같은 lifecycle 필요성이 확인되기 전에는 새 추상화를 추가하지 않는다.
- pretrained는 `weights=None` + 로컬 `.pth`의 `load_state_dict`로 주입한다. 로컬 자산이 없을 때 조용히 랜덤 초기화로 폴백하지 않는다.
- 실험 관리 도구(tensorboard, wandb)를 도입하지 않는다.
- 새 의존성 추가 전 기존 스택(torch, torchvision, torchmetrics)으로 가능한지 확인하고 사용자 승인을 받는다.
- `core/`는 anomaly 외 4개 task(classification, detection, segmentation, toy)가 공유한다. 변경 시 이들의 영향을 확인한다.

## 성능 판정 규칙

- **에이전트는 학습·평가를 임의로 장시간 실행하지 않는다. 실행 주체는 사용자다.**
- 구현을 마치면 사용자가 실행할 명령어를 제시하고, 피드백받은 결과를 anomalib reference와 비교한다.
- 차이가 나면 모델 구현 차이로 단정하지 않고 `PRD.md` §5.4 항목(preprocessing, optimizer, pretrained weight, normalization, threshold, metric, protocol)을 먼저 확인한다.
- 원인이 무엇이든 모델 코드는 수정하지 않는다. adapter 또는 boilerplate에서 해결한다.
- 동작 확인 스모크는 단일 카테고리 소규모로 한다.

## Common Contract and Agent Execution Rules

- P0·P1·P6은 메인 세션이 직접 수행한다. 모델 통합 Phase(P2, P4)는 모델별 에이전트에 위임할 수 있다.
- 모델 에이전트는 자기 `adapters/<model>.py`와 해당 config 외의 파일을 수정하지 않는다.
- 공통 코드(`core/`, `tasks/anomaly/adapter.py`, `cli/`, `bench/`) 수정 권한은 메인 세션만 갖는다. 하위 에이전트는 문제·재현 조건·최소 수정안·SPEC 조항을 담은 변경 요청만 반환한다.
- **`upstream/` 하위는 어떤 에이전트도 수정하지 않는다.** P1-T01의 import 경로 조정이 유일한 예외다.
- 공통 코드 변경 등급 C(계약 변경)는 사용자 승인 후에만 진행한다. 어떤 등급이든 공통 루프의 모델명 분기는 허용하지 않는다.
- 공통 코드가 바뀌면 완료된 모든 모델의 스모크를 재실행한다.
- Phase와 task는 `backlog.json`의 `depends_on`을 지켜 진행하고, 같은 `parallel_group`만 동시에 진행한다.

## Document Rules

- 개발 문서는 `docs/dev/v{major}.{minor}/`에 둔다. 문서 체인은 `BRIEF.md → PRD.md → SPEC.md → PLAN.md → backlog.json`이다.
- 구현 또는 프로젝트 내용이 변경되면 `SPEC.md → PLAN.md → backlog.json → PRD.md` 순서로 갱신한다.
- **`docs/refs/`는 읽기 전용이다.** 이전 분석 결과와 `comparison/` 비교 문서를 보존한다. 새로 분석하지 말고 `PLAN.md` §1.1 색인에서 필요한 절만 조회한다.
- 완료된 버전의 문서는 참조 전용으로 유지하며 사용자의 명시적 요청 없이는 수정하지 않는다.
- Phase 완료 상태는 `backlog.json`의 `status` 필드에서만 관리한다.
- 문서도 사용자 요청이 있으면 반대 벤더 CLI로 적대적 검증을 받으며, 검증 횟수는 Verification Attempt Limit을 따른다.

## Phase Execution Workflow

1. 해당 Phase task의 `scope`, `verification`, `completion_criteria`를 구현하고 검증한다. P3·P5는 사용자에게 실행 명령어를 제시하고 결과를 받는다.
2. 마지막 실질 구현자의 **반대 벤더 CLI**에 적대적 검증을 위임한다. Claude Code 구현은 Codex CLI가, Codex 구현은 Claude Sonnet headless CLI가 검토한다. 구현자가 중간에 바뀌면 마지막 실질 구현자를 기준으로 다시 정한다.
3. Critical 지적은 모두 수정하고 관련 검증을 재실행한다. Major와 Minor는 처리 여부와 근거를 기록한다. Critical을 수정했다면 같은 반대 벤더 검토를 한 번 더 실행해 해소를 확인한다.
4. `docs/dev/v{major}.{minor}/reviews/A{n}.md`에 구현자, 검토 모델, 대상 파일, 실행 일시, 심각도별 건수, 지적·재현 조건·관련 SPEC 조항·처리 상태를 기록한다. 유효하지 않은 지적의 반박 근거도 기록한다.
5. 변경 내용, 검증 결과, 검토 결과와 남은 위험을 사용자에게 보고하고 커밋 승인을 요청한다. 승인 전에는 커밋하지 않는다.

적대적 검증 필수 통과 Phase는 P1, P2, P4, P5다. 미해결 Critical이 있으면 다음 Phase로 진행하지 않는다. 필요한 반대 벤더 CLI를 사용할 수 없으면 오류와 대체 검증안을 사용자에게 보고하고 승인 없이 생략하지 않는다.

## Verification Attempt Limit

- 하나의 검증 대상(Phase 1개 또는 문서 1개)에 대한 검토 CLI 실행은 실패·오류·프롬프트 재작성·재검토를 모두 포함해 최대 3회로 제한한다.
- 지적 보완 후의 재검증도 이 3회에 포함된다. 보완 사이클을 별도로 세지 않는다.
- 3회를 소진하면 마지막 유효 검토 결과, 반영한 보완 내용, 미해결 지적과 남은 위험을 기록하고 사용자에게 보고한다.
- 3회 소진 시점에 미해결 Critical이 남아 있으면 다음 Phase로 진행하지 않고 사용자 판단을 요청한다.
- 실행 횟수와 각 회차의 결과(모델, 실행 일시, 유효 여부, 사유)를 검토 기록 문서에 남긴다.

## Adversarial Review Rules

- 검토자는 제품 소스 파일, 해당 Phase의 공격 초점, `spec_refs`만 사용한다. 구현 세션 대화, 구현 판단 근거, 문서, 설정, 실험 산출물, Git 상태·이력·원격 저장소, 셸 도구는 요청하거나 사용하지 않는다.
- 검토자는 파일을 수정하지 않는다. 지적은 `Critical / Major / Minor`, 정확한 재현 조건, 위반한 SPEC 조항을 포함해 심각도순으로 반환한다.
- 이 프로젝트의 우선 공격 축은 다음과 같다. 앞의 세 축이 이 프로젝트 고유의 것이며 가장 중요하다.
  - **upstream 원본 훼손** — import 경로 외 변경, 스타일 통일 명목의 수정 (CON-001)
  - **Lightning 잔재** — import·의존성·호출 경로, `lightning_model.py` 내용의 부적절한 이관 (CON-002)
  - **anomalib 동등성 이탈** — adapter의 호출 순서·loss 조합·정규화·map 합성이 원본과 어긋남
  - 공통 engine 오염 — `core/`나 공통 adapter로 새어 나온 모델 전용 개념 (NFR-005)
  - lifecycle hook 순서 — `on_fit_end`의 calibration→threshold 순서, `super()` 호출 누락 (SPEC §4.3)
  - 오프라인 위반 — `weights=`, `torch.hub`, 자동 다운로드, 조용한 랜덤 초기화 폴백 (CON-003, CON-004)
  - 학습/평가 누수 — train/valid/test 경계, `model.eval()`·`no_grad()`·metric 리셋 누락
  - frozen 파라미터 — teacher의 `eval()` 유지와 BatchNorm running stats, optimizer 포함 여부
  - 재현성 — seed 고정 범위, config·metric 기록이 실제 재현을 보장하는가
- Codex 검토는 `codex exec --model gpt-5.6-sol --sandbox read-only --cd "/mnt/d/projects/nampluskr/00_review/260820_defectvad-refactor"`를 사용한다.
- Claude Sonnet 검토는 `claude -p`에 `--model sonnet --safe-mode --allowedTools "Read,Glob,Grep" --disallowedTools "Edit,Write,Bash" --permission-mode dontAsk --max-turns 5 --output-format json --no-session-persistence`를 사용한다. 필요하면 `--max-budget-usd`로 비용 상한을 둔다.
- 모델 접근이 거부되면 기본 모델로 조용히 폴백하지 않고 오류와 대체안을 보고한다.
- 문서 검증 시에는 검토자가 대상 문서와 상위 문서, `AGENTS.md`를 읽기 위해 읽기 전용 셸 명령(`cat`, `sed -n` 등)을 사용하는 것을 허용한다.
- 검토 CLI 실행의 시간 제한은 기본 10분으로 설정한다.
- 전체 명령 예시는 `AGENTS.md`의 검토 명령 절을 참조한다.

## upstream 무결성 검증

`upstream/` 하위는 적대적 검증과 별개로 매 Phase 완료 시 기계적으로 확인한다. 검토 CLI의 판단에 의존하지 않는다.

- P1-T01에서 복사 시점의 원본과 diff해 import 문 외 차이가 없음을 확인하고 기록한다.
- 이후 Phase에서는 P1-T01 시점의 트리와 현재 트리를 비교해 변경이 없음을 확인한다.
- 변경이 발견되면 즉시 되돌린다. 변경이 필요해 보이는 상황은 adapter 또는 boilerplate 수정으로 해결한다.

## Commit and Push Rules

- 원격 저장소는 `https://github.com/nampluskr/defectvad-refactor`로 확정되었다(2026-08-20 사용자 확인).
- 로컬 저장소의 원격 연결과 첫 푸시는 사용자 승인 아래 수행한다. 그 전에는 커밋하지 않는다.
- 커밋은 하나의 완료된 Phase에 대응하며, Phase 번호와 핵심 변경 사항을 포함한다.
- 커밋 전에 해당 Phase의 검증과 `upstream/` 무결성 확인을 실행한다.
- 다른 작업의 변경 사항을 임의로 포함, 되돌리기, 삭제하지 않는다.
- 데이터셋, 백본 가중치, 체크포인트, 실험 산출물은 커밋하지 않는다.
- `docs/refs/`는 읽기 전용이다. 변경을 커밋에 포함하지 않는다.
