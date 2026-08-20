# PLAN — Anomaly Detection Integration on `cv_boilerplate`

> **저장소 참조 안내** — 본문의 `<repo>@<commit>:<path>#<symbol>` 표기는 아래 깃허브 저장소를 가리킨다.
>
> | 표기 | 깃허브 경로 |
> |---|---|
> | `defectvad@14879ea2` | https://github.com/nampluskr/defectvad |
> | `cv_boilerplate@65d5412b` | https://github.com/nampluskr/cv_boilerplate |
> | `roi-corner-detection-ver3@8ae989a8` | https://github.com/nampluskr/roi-corner-detection-ver3 |
>
> 특정 파일은 `https://github.com/nampluskr/<repo>/blob/<commit>/<path>` 로 조회한다.


문서 상태: Implementation Plan  
상위 문서: `BRIEF.md` > `PRD.md` > `SPEC.md`  
작성일: 2026-08-19  
현재 구현 확인 기준: `nampluskr/cv_boilerplate@71261cef`

## 1. Execution Summary

| Phase | Goal | Key References | Depends On | Parallel |
|---|---|---|---|---|
| P0 | 구현 대상과 reference baseline 고정 | FR-019, FR-022, CON-008; SPEC §11, §21 | - | Partial |
| P1 | 최소 anomaly foundation과 검증 가능한 dataset/metric 경계 구축 | FR-002, FR-013~FR-017; SPEC §3~§5, §12~§16 | P0 | Partial |
| P2 | STFPM 기반 첫 end-to-end vertical slice 완성 | FR-001, FR-003~FR-007, FR-010~FR-012; SPEC §6, §8~§10 | P1 | Partial |
| P3 | auxiliary 및 no-gradient lifecycle까지 공통 contract 확장 | FR-005, FR-006, FR-008, FR-021; SPEC §3.3, §7, §14 | P2 | Partial |
| P4 | 서로 다른 모델 lifecycle을 병렬로 통합하고 모델별 acceptance 수행 | FR-003~FR-008, FR-024; SPEC §6~§11, §17 | P3 | Yes, `MODEL-WAVE-1` |
| P5 | cross-model reference equivalence와 재현성 gate 통과 | FR-018~FR-020, FR-022, FR-025; SPEC §11~§15, §17 | P4 | Partial |
| P6 | offline 전체 lifecycle, 기존 task 회귀, provenance를 최종 검증 | NFR-003, NFR-006, NFR-013~NFR-014; SPEC §16~§19 | P5 | Partial |

이 순서는 첫 모델 전에 범용 workflow framework를 완성하려 하지 않는다. P1은 첫 vertical slice에 필요한 최소 contract와 검증 기반만 만들고, P2에서 실제 STFPM 흐름으로 경계를 검증한다. Auxiliary stream과 optimizer 없는 fitting은 P2 결과를 검토한 뒤 P3에서 추가하며, 이 contract가 안정된 이후에만 P4의 모델별 작업을 병렬화한다.

## 2. Specification Conflict

### SC-001 — 실제 구현 대상 source checkout 부재

- 확인 사실: 현재 작업 directory에는 `BRIEF.md`, `PRD.md`, `SPEC.md`만 있고 `cv_boilerplate` source와 Git metadata가 없다.
- 재검증 범위: 별도 read-only clone의 revision `71261cef`에서 `src/core/adapter.py`의 `TaskAdapter`, `src/core/engine.py`의 `Trainer`, `src/core/builders.py`, `src/cli/commands.py`, `src/bench/runner.py` 및 SPEC §22의 예상 경로가 존재함을 확인했다.
- 영향: local/uncommitted 차이와 실제 구현 branch를 알 수 없으므로 P1 이후의 정확한 변경 파일과 regression baseline을 확정할 수 없다.
- 처리: P0-T01에서 실제 checkout과 revision을 고정한다. 불일치가 발견되면 SPEC을 임의로 우회하지 않고 conflict report를 작성하여 사용자 승인을 받는다.

현재 확인한 API 이름과 SPEC 사이에는 그 밖의 명백한 conflict가 없다.

## 3. Pre-Implementation Decisions

| Decision | Blocks | Options | Recommended Direction | Reason |
|---|---|---|---|---|
| D-001 실제 source checkout/revision | P1~P6 | `71261cef` checkout / 다른 local revision 제공 | 실제 구현 대상 checkout을 workspace에 제공하고 commit 고정 | 현재 source가 없어 구현·회귀 검증 불가 |
| D-002 target anomalib revision | P0-T04 이후 | release tag / commit | source와 config가 함께 고정되는 commit hash | upstream diff와 reference 재현의 기준 |
| D-003 초기 모델 집합 | P2~P6 | STFPM+EfficientAD+PatchCore / 동등한 lifecycle 조합 | STFPM, EfficientAD, PatchCore | gradient, auxiliary/calibration, no-gradient를 최소 세 모델로 검증 |
| D-004 MVTec category 범위 | P0-T04, P5 | bottle / 일부 category / 전체 15 category | bottle로 pipeline을 고정하고 승인된 범위로 확장 | 초기 비용과 full acceptance 비용 분리 |
| D-005 validation/test split | P0-T04, P1-T04, P5 | strict disjoint / exact reference / 두 protocol 병행 | leakage-safe final과 exact-reference run을 별도 식별 | correctness 비교와 최종 평가 의미를 혼동하지 않음 |
| D-006 필수 metric | P1-T05, P5 | image/pixel AUROC / AUPRO·F1 추가 | target reference가 보고하는 metric을 필수화 | dependency와 tolerance가 metric 정의에 의존 |
| D-007 tolerance와 반복 횟수 | P5-T03 | 임의 고정 / 반복 baseline 기반 | pinned 환경의 반복 reference 결과로 승인 | NFR-012 충족 |
| D-008 upstream source transport | P2-T01, P4 | vendor / immutable sync / package dependency | runtime에 anomalib 전체를 넣지 않는 immutable source+diff 방식 | fidelity, license, offline 요구 동시 충족 |
| D-009 `timm` 허용 | STFPM 작업 | runtime dependency / narrow injection | target source diff와 offline 설치 비용을 함께 검토 후 승인 | algorithm fidelity와 최소 dependency trade-off |
| D-010 AUPRO 구현/dependency | P1-T05, P5 | verified port / existing library | reference fixture parity가 검증되는 최소 dependency | metric 의미 보존 |
| D-011 local asset inventory/checksum | P0-T03 이후 | 사용자 제공 inventory / 환경 audit | 실제 운영 환경 asset manifest 작성 | offline preflight와 provenance 필요 |
| D-012 checkpoint naming/compatibility | P1-T03, P2-T04 | 기존 best 갱신 / finalized 별도 이름 | 기존 CLI 기본을 보존하면서 inference-ready 의미 명시 | selection state와 finalized state 구분 |
| D-013 reference environment | P0-T04, P5 | 별도 pinned env / 외부 결과 import | 제품 runtime과 분리된 pinned environment | Lightning/Engine runtime 금지와 reference 실행 양립 |
| D-014 benchmark compute budget | P0-T05, P5 | full official / capped equivalent / 외부 baseline | fast smoke와 long acceptance profile 분리 | CI와 성능 검증의 목적 분리 |

D-001, D-002, D-003, D-005, D-006, D-011, D-013, D-014는 P0 exit 전에 해결해야 한다. 나머지는 해당 blocking task 전에 결정한다.

## Phase P0 — Reference Contract and Baseline Frozen

### Objective

구현 후 비교 조건이 변하지 않도록 실제 source, target anomalib source, dataset/split, metric, asset 및 실행 환경을 먼저 고정한다. 제품 runtime 구현은 하지 않으며 reference 환경과 small deterministic fixture를 준비한다.

### References

- BRIEF: §8 Benchmark의 역할, §9 G-06/G-09/G-10, §11 P-05/P-10
- PRD: FR-018~FR-023, NFR-001~NFR-003, NFR-006, NFR-012, CON-006~CON-012, GAP-009~GAP-013
- SPEC: §1.2, §5.3, §6.1~§6.2, §11, §13.4, §16, §17.3~§17.5, §19, §21

### Prerequisites

None.

### Scope

실제 구현 checkout과 revision을 확정하고, reference protocol manifest 및 local fixture를 만든다. 선택되지 않은 전체 모델이나 전체 MVTec category baseline은 만들지 않는다.

### Expected Code Impact

- 구현 source 변경 없음
- reference manifest, protocol decision record, environment lock 정보
- local-only smoke fixture와 asset inventory
- pinned anomalib reference raw result 및 반복 실행 요약

### Tasks

#### P0-T01 — Freeze implementation checkout and reconcile the specification

- Description: 실제 `cv_boilerplate` checkout, branch, commit 및 local changes를 확인하고 SPEC §22 경로/API와 대조한다. 차이는 `Specification Conflict`로 기록하고 승인 전 우회 설계를 금지한다.
- PRD refs: FR-022, NFR-002, CON-012, GAP-011
- SPEC refs: §1.2, §2, §21 OQ-001, §22
- Depends on: None
- Parallelizable: No
- Verification: commit identity, worktree state, component-path reconciliation report가 존재한다.

#### P0-T02 — Approve initial model, dataset, split, metric, and budget scope

- Description: D-003~D-007, D-014를 승인 가능한 protocol decision으로 고정한다. 추천 기본 조합은 STFPM, EfficientAD, PatchCore와 MVTec bottle이다.
- PRD refs: FR-009, FR-016, FR-019, NFR-001, NFR-012, CON-008~CON-010
- SPEC refs: §5.2~§5.3, §11.2~§11.4, §12.2, §21 OQ-003~OQ-007/OQ-014
- Depends on: P0-T01
- Parallelizable: No
- Verification: 승인된 model/category/split/metric/repeat/budget decision record가 완전하다.

#### P0-T03 — Inventory local assets and dependency constraints

- Description: dataset, backbone weight, auxiliary dataset, checksum, format 및 offline package availability를 조사한다. 누락 자산을 다운로드하지 않고 준비 책임과 경로를 기록한다.
- PRD refs: FR-008, FR-023, NFR-006, NFR-014, CON-006~CON-007
- SPEC refs: §6.4, §13.3, §16, §19, §21 OQ-009~OQ-011
- Depends on: P0-T02
- Parallelizable: Yes
- Parallel Group: `BASELINE-PREP`
- Verification: required/optional asset manifest와 dependency decision 후보가 작성되고 checksum 검증 가능하다.

#### P0-T04 — Pin upstream source, license, and reference environment

- Description: anomalib commit, source closure, reference config, license/notice, source transport 및 별도 reference environment를 고정한다. 제품 runtime과 reference runtime dependency를 분리한다.
- PRD refs: FR-003, FR-004, FR-019, FR-022, NFR-003, CON-002, CON-008, CON-011
- SPEC refs: §6.1~§6.2, §11.2, §17.5, §19, §21 OQ-002/OQ-008/OQ-013
- Depends on: P0-T02
- Parallelizable: Yes
- Parallel Group: `BASELINE-PREP`
- Verification: commit, source paths, license, diff baseline, config와 environment lock이 machine-readable하다.

#### P0-T05 — Capture smoke fixtures and repeated reference baseline

- Description: 승인된 category/model protocol로 fast fixture와 raw reference results를 생성하고 반복 분산을 기록한다. Full tolerance의 최종 승인은 P5에서 수행한다.
- PRD refs: FR-019, FR-020, FR-022, NFR-001, NFR-002, NFR-012, CON-008
- SPEC refs: §11.1~§11.4, §17.3~§17.4
- Depends on: P0-T03, P0-T04
- Parallelizable: No
- Verification: raw outputs, environment, resolved config, metric semantics, repeat summary와 protocol manifest가 연결된다.

#### P0-T06 — Validate baseline readiness

- Description: P0 산출물의 completeness와 offline availability를 검사하고 unresolved decision을 blocking 상태로 명시한다.
- PRD refs: FR-022, FR-023, NFR-002, NFR-011
- SPEC refs: §11.2, §13.4, §15, §16.4, §21
- Depends on: P0-T05
- Parallelizable: No
- Verification: D-001/002/003/005/006/011/013/014가 해결되고 reference manifest validation이 통과한다.

### Verification

- Unit: manifest schema, checksum, split membership validation
- Smoke: local fixture로 reference command가 완료됨
- Reference benchmark: 승인된 short/repeated baseline raw result 확보
- Regression: 해당 없음, 제품 source 미변경

### Exit Criteria

- 실제 implementation commit과 target anomalib commit이 고정된다.
- 초기 모델, category, split, metric 및 compute profile이 승인된다.
- required local asset이 식별되고 silent fallback 없이 검증 가능하다.
- reference raw result와 environment/protocol metadata가 연결된다.
- SC-001이 해결되거나 P1을 명시적으로 차단한다.

### Risks / Watch Items

- reference environment를 제품 dependency에 섞는 것
- test=validation reference와 leakage-safe final 결과를 같은 이름으로 기록하는 것
- baseline 완료 전에 tolerance를 임의로 선택하는 것

### Parallel Execution

- Parallel Group: `BASELINE-PREP`
- Workstreams: P0-T03 asset/dependency audit, P0-T04 source/reference environment pinning
- Shared Dependencies: P0-T02
- Merge/Gate requirement: 두 결과가 모두 P0-T05 protocol manifest에 병합되어야 한다.

## Phase P1 — Minimal Anomaly Foundation Ready

### Objective

첫 reference model을 end-to-end로 연결하는 데 필요한 최소 공통 contract만 구현한다. Existing registry, `TaskAdapter`, CLI, config, checkpoint container와 MVTec component를 확장하며 별도 anomaly engine이나 callback framework를 만들지 않는다.

### References

- BRIEF: §6, §11 P-01~P-04/P-06/P-08
- PRD: FR-002, FR-009, FR-012~FR-017, FR-021~FR-023, NFR-004/NFR-007/NFR-009~NFR-011, CON-003~CON-007/CON-009~CON-010
- SPEC: §3~§5, §12~§16, §17.1~§17.2, §22

### Prerequisites

P0 complete and SC-001 resolved.

### Scope

Common anomaly batch/output contract, MVTec split/transform, metric/post-processing state, checkpoint adapter state, local asset preflight 및 construction 중복 정리를 구현한다. Auxiliary loader와 optimizer-none 실행은 P3 범위다.

### Expected Code Impact

- Modify: `src/core/adapter.py`, `src/core/checkpoint.py`, `src/core/config.py`, `src/core/builders.py`, `src/cli/commands.py`, `src/bench/runner.py`
- Modify: `src/tasks/anomaly/adapter.py`, `dataset.py`, `transform.py`, `metric.py`, `postprocess.py`, `collate.py`
- Add: common anomaly contract/state tests, MVTec fixtures, asset/protocol validation tests
- Add/modify: approved anomaly base config, split and asset manifests

### Tasks

#### P1-T01 — Lock batch, output, and responsibility contracts with tests

- Description: SPEC §4 contract를 existing adapter/registry convention에 맞춰 executable tests로 고정한다. Dataset-specific layout이나 model name이 common metric path에 들어가지 않게 한다.
- PRD refs: FR-002, FR-013, FR-015, NFR-004, NFR-009
- SPEC refs: §3.1~§3.2, §4
- Depends on: P0-T06
- Parallelizable: No
- Verification: normal/anomalous batch와 score/map conversion contract tests가 통과한다.

#### P1-T02 — Unify component construction and validate anomaly configuration

- Description: CLI와 benchmark의 duplicated construction path를 현재 동작 보존 범위에서 공통화하고 anomaly protocol/asset fields를 fail-fast validation한다.
- PRD refs: FR-001, FR-002, FR-022, FR-023, FR-024, NFR-010~NFR-011
- SPEC refs: §2.2, §3.3, §13, §15
- Depends on: P1-T01
- Parallelizable: No
- Verification: CLI/benchmark가 같은 resolved components를 만들고 invalid config/asset은 construction 전에 실패한다.

#### P1-T03 — Add adapter and post-processing state to checkpoint round-trip

- Description: 기존 checkpoint container에 adapter/calibration state, global step, schema/protocol identity를 추가하고 backward policy를 적용한다. 평가 시 재보정하거나 model state에 억지로 넣지 않는다.
- PRD refs: FR-012, FR-017, FR-021, FR-022, NFR-002, NFR-011
- SPEC refs: §12.3, §14
- Depends on: P1-T01
- Parallelizable: Yes
- Parallel Group: `FOUNDATION-COMPONENTS`
- Verification: 새 adapter instance로 save/load 후 score normalization과 threshold state가 동일하다.

#### P1-T04 — Harden MVTec data and split protocol

- Description: parsing, zero mask, binary mask, image/mask geometry, stable sample ID와 materialized split manifest를 승인된 split policy에 맞춰 검증한다.
- PRD refs: FR-009, FR-014, FR-015, CON-009, CON-010
- SPEC refs: §5.1~§5.4
- Depends on: P1-T01
- Parallelizable: Yes
- Parallel Group: `FOUNDATION-COMPONENTS`
- Verification: MVTec fixture의 normal/defect/mask correspondence, disjointness 및 explicit shared-split flag가 통과한다.

#### P1-T05 — Implement approved metric and post-processing semantics

- Description: 승인된 image/pixel metric과 필요한 AUPRO/F1, smoothing, normalization, threshold ownership을 protocol 순서대로 구현·검증한다.
- PRD refs: FR-013, FR-016, FR-017, NFR-007, CON-009
- SPEC refs: §4.4~§4.5, §12, §19
- Depends on: P1-T01, P0-T05
- Parallelizable: Yes
- Parallel Group: `FOUNDATION-COMPONENTS`
- Verification: frozen fixtures에서 reference metric/input semantics와 state serialization이 일치한다.

#### P1-T06 — Enforce local asset preflight and provenance recording

- Description: existing offline guard와 strict local weight loader를 확장하여 dataset/weight manifest, checksum, consumer와 provenance를 검사·기록한다.
- PRD refs: FR-022, FR-023, NFR-006, NFR-011, CON-006~CON-007
- SPEC refs: §6.4, §13.3~§13.4, §15, §16
- Depends on: P0-T03, P1-T02
- Parallelizable: Yes
- Parallel Group: `FOUNDATION-COMPONENTS`
- Verification: missing/mismatched asset이 식별 가능한 오류로 실패하고 network call이나 random fallback이 없다.

#### P1-T07 — Integrate and gate the minimal foundation

- Description: component branches를 병합하고 toy/fixture model로 dataset→adapter→metric→checkpoint→predict path를 검증한다. 모델별 알고리즘을 이 task에 넣지 않는다.
- PRD refs: FR-010~FR-013, FR-021, NFR-007, NFR-013
- SPEC refs: §9, §10, §14.5, §17.1~§17.2
- Depends on: P1-T02, P1-T03, P1-T04, P1-T05, P1-T06
- Parallelizable: No
- Verification: integration tests, checkpoint new-process simulation, existing task focused regression이 통과한다.

### Verification

- Unit: contract, MVTec parsing/transform, metric, state, config/asset negative tests
- Integration: fixture dataset→adapter→metric→checkpoint→evaluate/predict
- Smoke: network blocked local fixture flow
- Regression: Classification/Segmentation/Detection focused construction and checkpoint tests

### Exit Criteria

- Approved batch/output/metric contract tests pass.
- MVTec mask alignment과 split manifest가 검증된다.
- Adapter/post-processing checkpoint round-trip이 새 instance에서 통과한다.
- CLI와 benchmark construction semantics가 일치한다.
- Common core/CLI에 anomaly model-name conditional이 없다.

### Risks / Watch Items

- 첫 모델 전에 auxiliary/no-gradient abstraction까지 확대하는 것
- post-processing state를 model parameter로 위장하는 것
- 기존 task checkpoint/config compatibility 손상

### Parallel Execution

- Parallel Group: `FOUNDATION-COMPONENTS`
- Workstreams: P1-T03 checkpoint state, P1-T04 dataset/split, P1-T05 metric/post-process, P1-T06 assets
- Shared Dependencies: P1-T01, P1-T02 일부
- Merge/Gate requirement: P1-T07이 shared contract와 regression을 통합 검증한다.

## Phase P2 — STFPM Vertical Slice Accepted

### Objective

승인된 첫 모델로 dataset→model→training→checkpoint→evaluation→prediction→metric→benchmark 전체 흐름을 완성한다. 기본 후보 STFPM은 현재 legacy evidence와 gradient teacher/student lifecycle을 이용해 최소 core extension의 적합성을 검증한다.

### References

- BRIEF: §3, §4, §5, §11 P-03/P-04/P-10
- PRD: FR-001, FR-003~FR-007, FR-010~FR-013, FR-019~FR-023, NFR-001~NFR-003/NFR-006/NFR-007, CON-001~CON-008/CON-011
- SPEC: §6, §7.2~§7.3, §8~§11, §14, §17

### Prerequisites

P1 complete; D-008, D-009, D-012 resolved.

### Scope

Pinned anomalib STFPM pure source, student-only optimization, model transform, output conversion, checkpoint state 및 per-model acceptance를 구현한다. EfficientAD auxiliary stream과 no-gradient fitting은 포함하지 않는다.

### Expected Code Impact

- Replace/adapt: `src/tasks/anomaly/models/stfpm.py` using pinned upstream source closure
- Add: STFPM-specific integration/adapter, loss/map modules as required, source/license manifest
- Add/modify: `configs/anomaly/stfpm.yaml`, local asset entries
- Add: pure model parity, lifecycle, command, checkpoint and reference comparison tests
- Generic core changes are forbidden unless promoted to a separate shared task and SPEC conflict review.

### Tasks

#### P2-T01 — Import and audit pinned STFPM pure-PyTorch source

- Description: target anomalib source closure를 최소 수정으로 가져오고 upstream checksum, license, imports와 diff를 기록한다.
- PRD refs: FR-003, FR-004, NFR-003, CON-004, CON-008, CON-011
- SPEC refs: §6.1~§6.2, §17.5, §18
- Depends on: P1-T07, P0-T04
- Parallelizable: Yes
- Parallel Group: `STFPM-SLICE`
- Verification: source manifest/diff audit와 pure model shape/parity fixtures가 통과한다.

#### P2-T02 — Implement STFPM integration and reference protocol

- Description: established anomaly contract로 teacher/student output, upstream loss, student-only parameter selection, optimizer/scheduler와 transform을 연결한다. Generic engine model-name 분기는 금지한다.
- PRD refs: FR-005~FR-007, FR-013, FR-017, FR-024
- SPEC refs: §6.3~§6.7, §7.2~§7.3, §8
- Depends on: P1-T07, P2-T01
- Parallelizable: No
- Verification: frozen parameter/optimizer/loss/map and resolved protocol tests가 reference 조건과 일치한다.

#### P2-T03 — Complete STFPM train and selection flow

- Description: approved validation policy로 학습·selection을 실행하고 final test가 selection/calibration에 사용되지 않음을 계측한다.
- PRD refs: FR-001, FR-009, FR-010, CON-009~CON-010
- SPEC refs: §7.4, §8, §9.1
- Depends on: P2-T02
- Parallelizable: No
- Verification: smoke training, selection checkpoint, leakage instrumentation과 metrics가 통과한다.

#### P2-T04 — Verify STFPM checkpoint, evaluate, and predict round-trip

- Description: inference-ready checkpoint를 새 process에서 복원하여 evaluate/predict output, score/map, threshold 및 visualization input을 비교한다.
- PRD refs: FR-010~FR-012, FR-017, FR-021, FR-023
- SPEC refs: §9, §10, §12.3, §14
- Depends on: P2-T03
- Parallelizable: No
- Verification: identical input의 pre/post-load output이 승인된 numerical tolerance 내에서 일치한다.

#### P2-T05 — Run STFPM per-model acceptance and minimal reference comparison

- Description: train/evaluate/predict/benchmark smoke와 pinned reference fixture comparison을 수행하고 protocol diff를 남긴다.
- PRD refs: FR-018~FR-020, FR-022, NFR-001~NFR-002, NFR-008
- SPEC refs: §11, §17.3~§17.4
- Depends on: P2-T04
- Parallelizable: No
- Verification: command matrix, raw metric, protocol diff, environment 및 provenance artifact가 완전하다.

#### P2-T06 — Review abstraction after the first vertical slice

- Description: STFPM 때문에 generic core에 들어간 책임을 검토하고 P3 capability가 실제 다른 lifecycle 요구에서만 도출되었는지 확인한다. 불필요한 model-specific core logic은 제거한다.
- PRD refs: FR-024, NFR-004~NFR-005, NFR-010, CON-005
- SPEC refs: §3.2~§3.3, §6.3, §7.1, §17.5
- Depends on: P2-T05
- Parallelizable: No
- Verification: purity scan과 architecture review에서 model-name/task-name branch 및 duplicated framework가 없다.

### Verification

- Unit: upstream parity, feature/output/loss, trainable parameter selection
- Integration: full train/checkpoint/evaluate/predict
- Smoke: MVTec approved category, offline local weights
- Reference benchmark: per-model frozen fixture comparison
- Regression: common engine focused existing-task tests

### Exit Criteria

- STFPM 공통 command matrix가 offline smoke에서 완료된다.
- Upstream source/diff/license가 추적된다.
- Student-only optimization과 reference transform이 resolved record에서 확인된다.
- New-process checkpoint round-trip과 calibrated prediction이 통과한다.
- Common engine에 STFPM 이름 분기가 없다.

### Risks / Watch Items

- 기존 simplified STFPM을 upstream source로 잘못 간주하는 것
- outer `model.train()`이 frozen teacher mode를 변경하는 것
- first-model convenience를 generic contract로 과잉 일반화하는 것

### Parallel Execution

- Parallel Group: `STFPM-SLICE`
- Workstreams: P2-T01 upstream source audit는 P1 완료 뒤 독립 준비 가능
- Shared Dependencies: P1 contract와 P0 reference manifest
- Merge/Gate requirement: P2-T02 전 source audit가 통과해야 하며 이후 flow는 순차 실행한다.

## Phase P3 — Heterogeneous Lifecycle Contract Stabilized

### Objective

P2에서 검증한 최소 경계를 깨지 않으면서 EfficientAD와 PatchCore가 실제로 요구하는 capability만 공통 contract에 추가한다. Optimizer 없는 fitting, step scheduler, validation preparation, named auxiliary loader와 model-specific state를 generic하게 검증한다.

### References

- BRIEF: §6.2, §11 P-06/P-08/P-10, §13
- PRD: FR-005, FR-006, FR-008, FR-021, FR-024, NFR-004/NFR-005/NFR-010/NFR-013, CON-001/CON-003/CON-005, OOS-006
- SPEC: §3.3, §5.5, §6.7~§6.8, §7, §8, §14, §17

### Prerequisites

P2 complete and abstraction review accepted.

### Scope

`TaskAdapter` extension, generic Trainer execution, builders/config/checkpoint를 확장한다. 특정 모델의 알고리즘, metric 또는 directory parsing은 core에 넣지 않는다.

### Expected Code Impact

- Modify: `src/core/adapter.py`, `engine.py`, `builders.py`, `config.py`, `checkpoint.py`
- Modify: shared anomaly adapter/state only where contract requires
- Add: lifecycle test doubles for gradient, auxiliary, collect/finalize and optimizer-none paths
- Add: engine purity and existing-task regression tests

### Tasks

#### P3-T01 — Add adapter-owned optimization specification and cadence

- Description: trainable parameter groups, optional optimizer/scheduler, step/epoch cadence와 gradient clipping을 existing builder/Trainer 경계에 추가한다.
- PRD refs: FR-005, FR-006, FR-024, NFR-004~NFR-005
- SPEC refs: §3.3, §6.7, §7.3, §8.1
- Depends on: P2-T06
- Parallelizable: No
- Verification: optimizer-present/none, selected parameter, step/epoch scheduler unit tests가 통과한다.

#### P3-T02 — Add validation preparation and finalize hook ordering

- Description: validation 직전/직후 optional hook과 fitting finalize 순서를 명시적으로 호출하며 test data 접근을 차단한다.
- PRD refs: FR-005, FR-009, FR-012, CON-009
- SPEC refs: §3.3, §7.4, §8.2, §9.1
- Depends on: P3-T01
- Parallelizable: Yes
- Parallel Group: `LIFECYCLE-CORE`
- Verification: hook-order trace와 split-access instrumentation이 expected sequence와 일치한다.

#### P3-T03 — Add named auxiliary loader construction and consumption

- Description: existing Dataset/Transform/DataLoader registry를 이용해 declared named auxiliary loaders를 만들고 adapter에 loader mapping으로 전달한다.
- PRD refs: FR-008, FR-023, NFR-006, CON-006
- SPEC refs: §3.3, §5.5, §7.2, §8.1, §13.2
- Depends on: P3-T01
- Parallelizable: Yes
- Parallel Group: `LIFECYCLE-CORE`
- Verification: missing, wrong-name, exhausted/restarted auxiliary stream과 offline asset tests가 통과한다.

#### P3-T04 — Complete heterogeneous lifecycle checkpoint state

- Description: lifecycle readiness, memory/statistics, normalization, global step 및 auxiliary progress에 필요한 adapter state를 generic checkpoint container로 round-trip한다.
- PRD refs: FR-012, FR-021, FR-022, NFR-002, NFR-011
- SPEC refs: §6.8, §7.5, §8.3, §14
- Depends on: P3-T02, P3-T03
- Parallelizable: No
- Verification: gradient/collect-finalize/auxiliary test double의 new-instance round-trip이 통과한다.

#### P3-T05 — Validate lifecycle capabilities without model-name branches

- Description: test doubles로 gradient, auxiliary, no-gradient collect/finalize를 실행하고 기존 task 회귀 및 purity scan을 수행한다.
- PRD refs: FR-005, FR-024, NFR-004/NFR-005/NFR-007/NFR-013, CON-005
- SPEC refs: §7.1, §17.1~§17.2, §17.5
- Depends on: P3-T04
- Parallelizable: No
- Verification: all lifecycle integration tests, static conditional scan and existing task regression pass.

#### P3-T06 — Freeze model-wave contract and merge gate

- Description: P4 model agents가 수정 가능한 model-specific 영역과 금지된 core 영역을 확정한다. 추가 core 요구는 독립 shared task로 승격하는 규칙을 기록한다.
- PRD refs: FR-024, NFR-005/NFR-010, CON-005
- SPEC refs: §3.1~§3.3, §6.3, §7.1, §18
- Depends on: P3-T05
- Parallelizable: No
- Verification: contract checklist, allowed/avoid paths와 merge gate가 model workstream template에 반영된다.

### Verification

- Unit: optimization spec, cadence, hooks, auxiliary loader, state schema
- Integration: three lifecycle test doubles
- Smoke: network-blocked auxiliary fixture
- Regression: all existing task engine/checkpoint/benchmark tests

### Exit Criteria

- Optimizer-none와 auxiliary lifecycle이 generic API로 실행된다.
- Validation preparation과 finalize hook order가 테스트로 고정된다.
- 모든 lifecycle state가 checkpoint round-trip된다.
- Generic engine/CLI에 EfficientAD, PatchCore 또는 anomaly model-name 분기가 없다.
- P4 allowed/avoid scope와 shared-change escalation rule이 승인된다.

### Risks / Watch Items

- callback/event framework로 범위를 확대하는 것
- auxiliary loader policy를 EfficientAD 이름에 결합하는 것
- scheduler cadence 변경으로 기존 task가 회귀하는 것

### Parallel Execution

- Parallel Group: `LIFECYCLE-CORE`
- Workstreams: P3-T02 hook ordering, P3-T03 auxiliary construction
- Shared Dependencies: P3-T01
- Merge/Gate requirement: P3-T04 state schema와 P3-T05 regression에서 병합 검증한다.
## Phase P4 — Model Lifecycle Wave 1 Accepted

### Objective

안정화된 contract 위에서 서로 다른 모델 integration을 독립 workstream으로 수행한다. 기본 승인 후보는 auxiliary/calibration lifecycle의 EfficientAD와 no-gradient memory-bank lifecycle의 PatchCore다.

### References

- BRIEF: §5, §6.2, §13, §11 P-03/P-04/P-10
- PRD: FR-003~FR-008, FR-010~FR-024, NFR-001~NFR-011, CON-001~CON-011
- SPEC: §5.4~§5.5, §6~§12, §14~§18

### Prerequisites

P3 complete; approved models and all model-specific assets available.

### Scope

각 workstream은 pure upstream source, model adapter/config/tests 및 per-model acceptance만 수정한다. Generic engine, common task contract, global config schema 수정은 금지하며 필요 시 P4-T03 shared change review로 중단·승격한다.

### Expected Code Impact

- Add/replace: model-specific pure source, loss/map and integration modules
- Add/modify: `configs/anomaly/efficientad.yaml`, PatchCore config, asset/source manifests
- Add: model-specific parity/lifecycle/command/checkpoint/reference tests
- Avoid: unassigned changes to `src/core/*`, common anomaly contract and unrelated models

### Tasks

#### P4-T01 — Integrate EfficientAD on the established contract

- Description: pinned pure-PyTorch EfficientAD source를 최소 수정으로 통합하고 teacher statistics, student/AE optimization, Imagenette auxiliary stream, input normalization, validation quantile, step scheduler 및 state를 구현한다. Generic core insufficiency가 확인되면 독자 수정하지 않고 P4-T03에 보고한다.
- PRD refs: FR-003~FR-008, FR-010~FR-013, FR-017, FR-021~FR-024
- SPEC refs: §5.4~§5.5, §6, §7.2~§7.5, §8~§10, §14, §18
- Depends on: P3-T06
- Parallelizable: Yes
- Parallel Group: `MODEL-WAVE-1`
- Verification: pure parity, auxiliary lifecycle, transform/batch, train/evaluate/predict, checkpoint state, offline smoke and minimal reference comparison pass.

#### P4-T02 — Integrate PatchCore on the established contract

- Description: pinned pure-PyTorch PatchCore source를 최소 수정으로 통합하고 optimizer-none feature collection, coreset/memory-bank finalize, anomaly output 및 serialized state를 구현한다. Generic core insufficiency가 확인되면 독자 수정하지 않고 P4-T03에 보고한다.
- PRD refs: FR-003~FR-006, FR-010~FR-013, FR-017, FR-021~FR-024
- SPEC refs: §6, §7.2~§7.5, §8~§10, §14, §18
- Depends on: P3-T06
- Parallelizable: Yes
- Parallel Group: `MODEL-WAVE-1`
- Verification: pure parity, optimizer-none collect/finalize, train/evaluate/predict, checkpoint memory state, offline smoke and minimal reference comparison pass.

#### P4-T03 — Adjudicate shared contract change requests

- Description: model agents가 입증한 공통 capability gap만 별도 shared change로 검토한다. 승인 시 core 변경과 전체 lifecycle regression을 한곳에서 수행하고 P4-T01/P4-T02를 재개한다. 요청이 없으면 no-op gate로 기록한다.
- PRD refs: FR-024, NFR-004/NFR-005/NFR-010/NFR-013, CON-005
- SPEC refs: §3.3, §7.1, §17.2, §17.5
- Depends on: P3-T06
- Parallelizable: No
- Verification: request별 two-model evidence, SPEC compatibility, purity and full regression 결과가 존재하거나 no-request가 기록된다. P4-T01/P4-T02는 승인된 shared change가 필요한 동안 blocked 상태를 유지한다.

#### P4-T04 — Run per-model acceptance matrix

- Description: STFPM, EfficientAD, PatchCore 각각에 대해 smoke, command lifecycle, state round-trip, missing asset, minimal reference comparison과 provenance를 동일 checklist로 검증한다.
- PRD refs: FR-001, FR-010~FR-023, NFR-001/NFR-002/NFR-006~NFR-008/NFR-011
- SPEC refs: §9~§17
- Depends on: P4-T01, P4-T02, P4-T03
- Parallelizable: No
- Verification: model별 acceptance artifact와 실패 진단이 완전하며 shared contract regression이 통과한다.

#### P4-T05 — Freeze model integration wave

- Description: model workstream이 공통 engine을 복제하지 않았고 upstream diff/license 및 extension cost가 요구사항을 만족하는지 review한다.
- PRD refs: FR-003, FR-004, FR-024, NFR-003~NFR-005/NFR-010, CON-004~CON-005/CON-011
- SPEC refs: §6.1~§6.3, §17.5, §18
- Depends on: P4-T04
- Parallelizable: No
- Verification: source audit, architecture purity, duplicate framework scan and change-scope review pass.

### Verification

- Unit: pure model parity, model-specific loss/output/state
- Integration: EfficientAD auxiliary lifecycle and PatchCore collect/finalize lifecycle
- Smoke: per-model offline command matrix
- Reference benchmark: per-model minimal frozen comparison
- Regression: STFPM and existing task suites after merge

### Exit Criteria

- EfficientAD와 PatchCore workstream의 model-specific tests와 command matrix가 통과한다.
- 각 모델의 state가 새 process checkpoint round-trip에서 복원된다.
- 각 모델의 upstream source/diff/license와 asset provenance가 완전하다.
- Shared core change는 P4-T03을 거쳤으며 model agent의 독자 core redesign이 없다.
- STFPM 포함 세 lifecycle의 common workflow가 유지된다.

### Risks / Watch Items

- 두 agent가 같은 core/config schema를 독립 수정하는 것
- EfficientAD teacher quantile에 test data를 사용하는 것
- PatchCore memory bank가 checkpoint에서 누락되는 것

### Parallel Execution

- Parallel Group: `MODEL-WAVE-1`
- Workstreams: P4-T01 EfficientAD, P4-T02 PatchCore
- Shared Dependencies: P3-T06 contract, P0 assets/reference fixtures
- Merge/Gate requirement: core 변경 요청은 P4-T03에서 단일 shared task로 처리하고 P4-T04에서 통합 검증한다.

## Phase P5 — Cross-Model Reference Equivalence Proven

### Objective

Benchmark를 단순 leaderboard가 아니라 integration correctness gate로 완성한다. 동일하거나 명시적으로 동등한 protocol에서 pinned anomalib reference와 각 integration을 비교하고 tolerance, 반복 재현성, 차이 진단 및 실패 격리를 검증한다.

### References

- BRIEF: §8, §9 G-06/G-09, §11 P-05/P-07
- PRD: FR-018~FR-020, FR-022, FR-025, NFR-001/NFR-002/NFR-008/NFR-012, CON-008~CON-010, AC-008/AC-009/AC-013~AC-015/AC-020
- SPEC: §11, §12, §13.4, §15.2, §17.4

### Prerequisites

P4 complete; long benchmark assets and compute budget approved.

### Scope

Reference-equivalence manifest, protocol diff, repeated acceptance, cross-model summary와 failure isolation을 구현·실행한다. 모델 알고리즘 tuning이나 새 algorithm 연구는 하지 않는다.

### Expected Code Impact

- Modify: `src/bench/runner.py`, `control.py`, `leaderboard.py` and common construction path as already designed
- Add: reference manifest/diff schema, benchmark profiles, result diagnostics and validation scripts/tests
- Add: approved long-run configs and raw/result artifacts

### Tasks

#### P5-T01 — Implement reference manifest and protocol diff validation

- Description: model/source/config/dataset/split/transform/optimizer/weight/post-process/metric/environment identity를 기록하고 reference와 integration 차이를 구조화한다.
- PRD refs: FR-018~FR-020, FR-022, NFR-002/NFR-008
- SPEC refs: §11.2~§11.3, §13.4
- Depends on: P4-T05
- Parallelizable: Yes
- Parallel Group: `BENCH-PREP`
- Verification: intentional protocol mismatch가 정확한 field diff와 approval state를 생성한다.

#### P5-T02 — Harden benchmark isolation and result completeness

- Description: 한 model/category 실패가 독립 실행 결과를 손상시키지 않도록 status, error, unfinished scope와 artifact를 보존한다.
- PRD refs: FR-018, FR-025, NFR-008, AC-020
- SPEC refs: §11.5, §15.2
- Depends on: P4-T05
- Parallelizable: Yes
- Parallel Group: `BENCH-PREP`
- Verification: injected failure 후 성공 row, failure row, diagnostics와 leaderboard integrity가 보존된다.

#### P5-T03 — Approve empirical tolerance and reproducibility envelope

- Description: P0 pinned reference 반복과 target environment 반복으로 metric tolerance와 repeat count를 확정하고 acceptance 전에 기록한다.
- PRD refs: FR-019, NFR-001/NFR-002/NFR-012, CON-008
- SPEC refs: §11.4, §17.4, §21 OQ-007
- Depends on: P5-T01, P0-T05
- Parallelizable: No
- Verification: raw repeats, summary statistics, environment and approved tolerance record가 연결된다.

#### P5-T04 — Run per-model long reference acceptance

- Description: 승인된 모델 각각에 대해 pinned reference와 integration을 동일/equivalent protocol로 반복 실행한다.
- PRD refs: FR-019, FR-020, NFR-001/NFR-002/NFR-012, AC-008/AC-009/AC-014
- SPEC refs: §11.4, §17.4
- Depends on: P5-T02, P5-T03
- Parallelizable: Yes
- Parallel Group: `REFERENCE-RUNS`
- Verification: 각 모델 metric이 승인 tolerance를 통과하거나 complete protocol diagnosis를 남긴다.

#### P5-T05 — Validate cross-model controlled benchmark

- Description: 동일 category/control envelope에서 모델별 결과, protocol exception, resource profile과 failure status를 요약한다. 모델 고유 reference config를 억지로 동일 hyperparameter로 만들지 않는다.
- PRD refs: FR-018~FR-020, FR-025, NFR-008
- SPEC refs: §11.1~§11.5
- Depends on: P5-T04
- Parallelizable: No
- Verification: machine-readable results, human summary, protocol diff links and failure isolation evidence가 완전하다.

### Verification

- Unit: manifest/diff/result schema
- Integration: injected failure and resume/overwrite behavior
- Reference benchmark: per-model repeated acceptance and controlled cross-model summary
- Regression: existing benchmark workflow and leaderboard

### Exit Criteria

- 각 승인 모델의 reference comparison이 tolerance를 통과하거나 승인 가능한 protocol diagnosis를 남긴다.
- Tolerance가 acceptance run 전에 반복 evidence로 고정된다.
- Split 공유/leakage 상태가 결과에 명시된다.
- 한 실행 실패가 다른 결과를 삭제하거나 성공으로 오인시키지 않는다.
- Fast CI profile과 long reference profile이 분리된다.

### Risks / Watch Items

- 동일 hyperparameter 강제로 reference semantics를 훼손하는 것
- 실패 모델을 leaderboard에서 조용히 누락하는 것
- 비결정성 차이를 algorithm 오류 또는 반대로 단정하는 것

### Parallel Execution

- Parallel Groups: `BENCH-PREP`, `REFERENCE-RUNS`
- Workstreams: manifest/diff와 failure isolation 준비, 이후 모델별 long reference run
- Shared Dependencies: frozen reference protocol and P4 model acceptance
- Merge/Gate requirement: P5-T03 tolerance 승인 후에만 P5-T04 acceptance를 판정하고 P5-T05에서 통합한다.

## Phase P6 — Offline, Regression, and Release Evidence Complete

### Objective

승인된 모델과 기존 task 전체에서 offline, dependency, source purity, checkpoint 및 regression evidence를 최종 확인한다. 구현 범위와 운영 절차를 문서화하되 새로운 architecture나 모델을 추가하지 않는다.

### References

- BRIEF: §9 G-07~G-10, §10, §11
- PRD: NFR-003~NFR-011/NFR-013~NFR-014, CON-001~CON-007/CON-011, AC-003/AC-011/AC-012/AC-016~AC-019
- SPEC: §15~§19, §22

### Prerequisites

P5 complete.

### Scope

Network-blocked full lifecycle, negative asset tests, full regression, static purity/license audit와 final evidence index를 완료한다.

### Expected Code Impact

- Tests/scripts: offline lifecycle, purity/import/download scan, full task regression, reproducibility checks
- Documentation/artifacts: supported model protocol, asset preparation, provenance and acceptance evidence index
- Dependency files: only already approved minimal runtime/reference dependency separation

### Tasks

#### P6-T01 — Run network-blocked lifecycle and asset failure matrix

- Description: 모든 승인 모델의 train/evaluate/predict/benchmark를 local assets로 실행하고 dataset/weight/auxiliary asset 누락·불일치 실패를 검증한다.
- PRD refs: FR-023, NFR-006/NFR-011, CON-006~CON-007, AC-011~AC-012
- SPEC refs: §15, §16, §17.2
- Depends on: P5-T05
- Parallelizable: Yes
- Parallel Group: `FINAL-VALIDATION`
- Verification: network attempt 없음, full lifecycle 성공, negative cases의 actionable error와 no fallback 확인.

#### P6-T02 — Run full existing-task and benchmark regression

- Description: Classification, Segmentation, Detection, toy/anomaly와 common benchmark의 승인 suite를 실행한다.
- PRD refs: NFR-004/NFR-005/NFR-013, AC-018
- SPEC refs: §17.2
- Depends on: P5-T05
- Parallelizable: Yes
- Parallel Group: `FINAL-VALIDATION`
- Verification: full regression suite passes or every pre-existing failure is independently evidenced.

#### P6-T03 — Audit runtime purity, dependencies, source, and licenses

- Description: Lightning/anomalib Engine import, implicit download API, model-name core branch, vendored diff, notice와 unnecessary dependency를 검사한다.
- PRD refs: NFR-003/NFR-004/NFR-014, CON-001~CON-002/CON-004~CON-005/CON-011, AC-002~AC-003/AC-017
- SPEC refs: §16.3, §17.5, §19
- Depends on: P5-T05
- Parallelizable: Yes
- Parallel Group: `FINAL-VALIDATION`
- Verification: static scans, dependency review, source checksum/diff and attribution audit pass.

#### P6-T04 — Assemble final acceptance and reproducibility evidence

- Description: model별 pure/integration/dataset/metric/reference 결과와 protocol, asset, environment, unresolved limitations를 하나의 index로 연결한다.
- PRD refs: FR-022, NFR-002/NFR-007/NFR-008, AC-013/AC-019
- SPEC refs: §11.2, §13.4, §17, §22
- Depends on: P6-T01, P6-T02, P6-T03
- Parallelizable: No
- Verification: every acceptance criterion has linked machine-readable and human-readable evidence.

#### P6-T05 — Release gate review

- Description: BRIEF/PRD/SPEC 범위, non-goals, remaining decisions와 all phase exit criteria를 검토하여 완료 또는 명시적 block을 판정한다.
- PRD refs: CON-012, OOS-001~OOS-011
- SPEC refs: §1.4, §18, §21, §22
- Depends on: P6-T04
- Parallelizable: No
- Verification: requirement coverage, non-goal audit and release decision record are complete.

### Verification

- Unit/Integration: asset negative tests and checkpoint/state suites
- Smoke: all model command matrix with network blocked
- Reference benchmark: P5 evidence revalidated, not rerun without reason
- Regression: full existing task and common benchmark suite
- Static: dependency/import/download/model-branch/source/license scans

### Exit Criteria

- Offline full lifecycle and missing-asset failure matrix pass.
- Existing Classification/Segmentation/Detection and benchmark regression pass.
- Product runtime has no Lightning/anomalib Engine dependency or implicit download.
- Upstream source/diff/license and minimal dependency audit pass.
- Every in-scope acceptance criterion links to evidence; unresolved items are explicit blockers, not silent omissions.

### Risks / Watch Items

- reference-only dependencies leaking into product runtime
- long benchmark를 fast CI에 포함해 검증이 비현실적으로 되는 것
- documentation task에서 미승인 기능을 새 scope로 추가하는 것

### Parallel Execution

- Parallel Group: `FINAL-VALIDATION`
- Workstreams: P6-T01 offline, P6-T02 regression, P6-T03 purity/dependency audit
- Shared Dependencies: P5-T05
- Merge/Gate requirement: P6-T04가 세 evidence stream을 통합하고 P6-T05가 최종 판정한다.

## 4. Requirement Coverage Matrix

### 4.1 Functional Requirements

| PRD ID | SPEC Section | PLAN Phase/Task | Verification |
|---|---|---|---|
| FR-001 | §2.1, §8~§11 | P1-T02, P2-T03~T05, P4-T04 | command matrix |
| FR-002 | §2.3, §4.1 | P1-T01~T02 | registry/construction tests |
| FR-003 | §6.1~§6.2 | P0-T04, P2-T01, P4-T01~T02 | source manifest/parity |
| FR-004 | §6.1, §6.3 | P2-T01, P4-T01~T02/T05 | upstream diff audit |
| FR-005 | §3.3, §7, §8 | P2-T02, P3-T01~T05, P4-T01~T02 | heterogeneous lifecycle tests |
| FR-006 | §6.7, §7.3 | P2-T02, P3-T01, P4-T01~T02 | parameter/cadence tests |
| FR-007 | §5.4, §13.2 | P2-T02, P4-T01 | resolved transform tests |
| FR-008 | §5.5, §7.2 | P0-T03, P3-T03, P4-T01 | auxiliary stream tests |
| FR-009 | §5.3, §7.4 | P0-T02, P1-T04, P2-T03 | leakage/split tests |
| FR-010 | §9 | P1-T07, P2-T03~T04, P4-T04 | checkpoint evaluation |
| FR-011 | §10 | P1-T07, P2-T04, P4-T04 | file/directory prediction |
| FR-012 | §9, §10, §14 | P1-T03/T07, P2-T04, P3-T04 | calibrated round-trip |
| FR-013 | §4.2~§4.4 | P1-T01/T05, P2-T02, P4-T04 | output semantics tests |
| FR-014 | §5.2, §5.4 | P1-T04 | MVTec fixture tests |
| FR-015 | §4.4, §5 | P1-T01/T04 | dataset-independence contract |
| FR-016 | §12.2 | P0-T02, P1-T05, P5-T04 | metric fixture/parity |
| FR-017 | §4.5, §12 | P1-T03/T05, P2-T04, P4-T04 | post-process state tests |
| FR-018 | §11 | P2-T05, P5-T01~T05 | benchmark artifacts |
| FR-019 | §11.4 | P0-T04~T05, P2-T05, P5-T03~T04 | paired reference runs |
| FR-020 | §11.2~§11.3 | P0-T02, P2-T05, P5-T01/T04~T05 | protocol diff |
| FR-021 | §14 | P1-T03, P2-T04, P3-T04, P4-T04 | new-process round-trip |
| FR-022 | §13.4, §14, §16.4 | P0-T01/T03~T06, P1-T03/T06, P5-T01, P6-T04 | artifact completeness |
| FR-023 | §6.4, §15~§16 | P0-T03, P1-T06, P6-T01 | offline/negative asset tests |
| FR-024 | §3.2, §6.3 | P1-T02, P2-T02/T06, P3-T05~T06, P4-T05 | integration change review |
| FR-025 | §11.5, §15.2 | P5-T02/T05 | injected failure test |

### 4.2 Non-Functional Requirements

| PRD ID | SPEC Section | PLAN Phase/Task | Verification |
|---|---|---|---|
| NFR-001 | §11.4, §17.4 | P0-T05, P2-T05, P5-T03~T04 | approved tolerance benchmark |
| NFR-002 | §5.3, §14, §17.4 | P0-T01/T05, P1-T03, P3-T04, P5-T03~T04, P6-T04 | repeat/round-trip evidence |
| NFR-003 | §6.1~§6.2, §17.5 | P0-T04, P2-T01, P4-T05, P6-T03 | source diff audit |
| NFR-004 | §3.2, §17.5 | P1-T01, P2-T06, P3-T05, P6-T03 | purity scan |
| NFR-005 | §3.3, §7.1 | P2-T06, P3-T01/T05~T06, P6-T02 | lifecycle extension tests |
| NFR-006 | §16 | P0-T03, P1-T06, P4-T04, P6-T01 | network-blocked lifecycle |
| NFR-007 | §17 | P1-T01/T05/T07, P3-T05, P6-T04 | layered evidence |
| NFR-008 | §11.3, §15.2 | P2-T05, P5-T01~T05, P6-T04 | diagnostics/artifacts |
| NFR-009 | §3.2, §4.4, §5 | P1-T01/T04 | alternate contract review |
| NFR-010 | §3, §6.3, §18 | P1-T02, P2-T06, P3-T06, P4-T05 | responsibility review |
| NFR-011 | §15 | P0-T06, P1-T02/T06, P3-T04, P6-T01 | fail-fast negative tests |
| NFR-012 | §11.4, §17.4 | P0-T02/T05, P5-T03~T04 | repeated statistics |
| NFR-013 | §17.2 | P1-T07, P3-T05, P4-T03, P6-T02 | full regression |
| NFR-014 | §19 | P0-T03, P6-T03 | dependency audit |

### 4.3 Constraints

| PRD ID | SPEC Section | PLAN Phase/Task | Verification |
|---|---|---|---|
| CON-001 | §3.3, §8 | P3-T01~T05, P6-T03 | runtime/import inspection |
| CON-002 | §6.2, §17.5 | P0-T04, P6-T03 | Lightning/Engine scan |
| CON-003 | §2, §3.1 | P1-T02~T03, P3-T01~T04 | lifecycle ownership review |
| CON-004 | §4.5, §6.1 | P2-T01~T02, P4-T01~T02, P6-T03 | algorithm parity/diff |
| CON-005 | §3.2, §17.5 | P2-T06, P3-T05~T06, P4-T03/T05, P6-T03 | model-name conditional scan |
| CON-006 | §16 | P0-T03, P1-T06, P3-T03, P6-T01 | offline tests |
| CON-007 | §6.4, §15 | P0-T03, P1-T06, P6-T01 | no-fallback tests |
| CON-008 | §6.1, §11.2 | P0-T02/T04~T05, P5-T03 | pinned manifest |
| CON-009 | §5.3, §7.4 | P0-T02, P1-T04~T05, P2-T03 | split-access tests |
| CON-010 | §5.2~§5.3 | P0-T02, P1-T04, P5-T01 | explicit split manifest |
| CON-011 | §6.2, §17.5 | P0-T04, P2-T01, P4-T05, P6-T03 | license/source audit |
| CON-012 | §1, §21 | P0-T01/T06, P6-T05 | conflict/release review |

### 4.4 Acceptance Criteria

| PRD ID | SPEC Section | PLAN Phase/Task | Verification |
|---|---|---|---|
| AC-001 | §8~§11 | P2-T03~T05, P4-T04 | model command matrix |
| AC-002 | §6.1~§6.2 | P2-T01, P4-T05, P6-T03 | source audit |
| AC-003 | §6.2, §19 | P6-T03 | runtime/import scan |
| AC-004 | §7 | P3-T05, P4-T01~T04 | gradient/aux/no-gradient matrix |
| AC-005 | §5.4~§6.7, §13 | P2-T02, P4-T01~T02, P5-T01 | resolved protocol inspection |
| AC-006 | §5 | P1-T04 | MVTec data tests |
| AC-007 | §4.4, §12 | P1-T05, P5-T04 | metric parity |
| AC-008 | §11.4, §17.4 | P5-T03~T04 | tolerance acceptance |
| AC-009 | §11.3 | P2-T05, P5-T01/T04 | protocol diagnosis |
| AC-010 | §14.5 | P1-T03, P2-T04, P3-T04, P4-T04 | new-process parity |
| AC-011 | §16 | P6-T01 | network-blocked matrix |
| AC-012 | §15 | P1-T06, P6-T01 | missing asset failures |
| AC-013 | §11.2, §13.4 | P0-T05, P5-T01, P6-T04 | artifact schema |
| AC-014 | §11.4 | P5-T03~T04 | repeated runs |
| AC-015 | §5.3, §7.4 | P1-T04, P2-T03, P5-T01 | leakage instrumentation |
| AC-016 | §3.2~§3.3, §6.3 | P3-T06, P4-T05 | integration change review |
| AC-017 | §3.2, §17.5 | P2-T06, P3-T05, P6-T03 | purity scan |
| AC-018 | §17.2 | P6-T02 | full regression |
| AC-019 | §17 | P4-T04, P6-T04 | evidence matrix |
| AC-020 | §11.5, §15.2 | P5-T02 | injected failure |

### 4.5 Current Gap Closure

| PRD ID | SPEC Section | PLAN Phase/Task | Verification |
|---|---|---|---|
| GAP-001 | §6.1~§6.3 | P0-T04, P2-T01, P4-T01~T02 | upstream source/diff |
| GAP-002 | §3.3, §7.3 | P3-T01/T05, P4-T02 | optimizer-none test |
| GAP-003 | §6.7 | P2-T02 | student-only optimizer test |
| GAP-004 | §5.4 | P4-T01 | EfficientAD input protocol |
| GAP-005 | §5.5, §7.2 | P3-T03, P4-T01 | auxiliary loader test |
| GAP-006 | §7.2~§7.4 | P3-T02/T05, P4-T02 | collect/finalize test |
| GAP-007 | §10, §12.3, §14 | P1-T03, P2-T04, P3-T04 | state round-trip |
| GAP-008 | §12.2 | P0-T02, P1-T05, P5-T04 | metric parity |
| GAP-009 | §5.2~§5.3, §21 | P0-T02, P1-T04 | approved split manifest |
| GAP-010 | §11 | P0-T05, P5-T01~T05 | equivalence artifacts |
| GAP-011 | §11.2, §13.4 | P0-T01/T03~T05, P6-T04 | provenance completeness |
| GAP-012 | §6.4, §19, §21 | P0-T03~T04, P6-T03 | dependency/asset decision |
| GAP-013 | §11.4, §17.4 | P0-T05, P5-T03~T04 | repeated baseline |

### 4.6 Out-of-Scope Guards

| PRD ID | PLAN Treatment |
|---|---|
| OOS-001 | anomalib Engine/CLI/callback 재구현 task 없음; P6-T03에서 runtime import 차단 |
| OOS-002 | 모델 정확도 개선 연구 task 없음; P5는 protocol equivalence만 진단 |
| OOS-003 | P2/P4 source diff gate로 대규모 rewrite 금지 |
| OOS-004 | legacy trainer/evaluator/factory 이식 task 없음 |
| OOS-005 | anomalib API/checkpoint compatibility task 없음 |
| OOS-006 | P3에서 heterogeneous lifecycle을 지원하고 단일 training step 강제 금지 |
| OOS-007 | D-003에서 승인한 초기 모델만 계획 |
| OOS-008 | MVTec만 초기 dataset task로 계획 |
| OOS-009 | 원격 tracking, registry, cloud orchestration task 없음 |
| OOS-010 | P5 tolerance는 numerical equivalence이며 bitwise 동일성 아님 |
| OOS-011 | P1/P6은 local asset preflight만 구현하며 download service 없음 |

## 5. SPEC Implementation Coverage

| SPEC Section | PLAN Phase/Task | Notes |
|---|---|---|
| §1 Purpose and Scope | P0-T01/T06, P6-T05 | hierarchy and scope gates |
| §2 Current Architecture Summary | P0-T01, P1-T02 | actual checkout reconciliation |
| §3 Target Architecture | P1-T01~T03, P2-T06, P3, P4-T03 | minimal extension and purity |
| §4 Anomaly Detection Task Contract | P1-T01/T05, P2-T02 | executable contract tests |
| §5 Dataset Integration | P0-T02~T03, P1-T04, P3-T03 | MVTec, splits, auxiliary data |
| §6 Model Integration | P0-T04, P2-T01~T02, P4-T01~T02 | upstream source and adapters |
| §7 Model Lifecycle Variations | P2, P3, P4 | staged lifecycle proof |
| §8 Training Flow | P2-T03, P3-T01~T03, P4 | gradient/aux/no-gradient |
| §9 Evaluation Flow | P1-T07, P2-T04, P4-T04 | checkpoint-only evaluation |
| §10 Prediction Flow | P1-T07, P2-T04, P4-T04 | calibrated outputs |
| §11 Benchmark/Reference Equivalence | P0-T04~T05, P2-T05, P5 | early and cross-model gates |
| §12 Metrics/Post-processing | P1-T03/T05, P5-T04 | ownership and parity |
| §13 Configuration | P0-T02~T03, P1-T02/T06, P5-T01 | validated resolved protocol |
| §14 Checkpoint/State | P1-T03, P2-T04, P3-T04, P4-T04 | new-process round-trip |
| §15 Errors | P1-T06, P5-T02, P6-T01 | fail-fast and isolation |
| §16 Offline | P0-T03, P1-T06, P6-T01/T03 | local assets only |
| §17 Testing | every phase; P6-T04 | layered evidence |
| §18 Legacy Migration | P2-T01, P4-T05, P6-T03 | evidence only, no framework port |
| §19 Dependencies | P0-T03~T04, P1-T05, P6-T03 | minimum approved dependencies |
| §20 PRD Traceability Matrix | §4 of this PLAN, P6-T04 | complete mapping |
| §21 Open Questions | §3, P0-T01~T06 | decisions tied to blocking phase |
| §22 Implementation Impact | P1~P6 Expected Code Impact | paths revalidated after D-001 |

## 6. Plan Self-Review

- BRIEF 범위 밖의 algorithm research, legacy framework, MLOps 또는 download service phase가 없다.
- 모든 FR/NFR/CON/AC/GAP ID가 최소 한 task와 verification에 연결되며 OOS는 별도 guard로 처리한다.
- 첫 모델 전 P1은 contract, dataset, metric, state와 asset validation으로 제한한다.
- STFPM vertical slice 후에만 auxiliary/no-gradient capability를 공통화한다.
- Model-specific workstream은 `src/core/*`와 common contract를 독자 수정할 수 없다.
- EfficientAD와 PatchCore는 stable contract 이후 `MODEL-WAVE-1`에서 병렬화한다.
- Per-model reference comparison은 P2/P4에서 수행하고 cross-model long gate는 P5에서 수행한다.
- Fast smoke와 long reference benchmark를 분리한다.
- Offline/local asset, checkpoint model-specific state, leakage 및 existing-task regression이 exit criteria에 포함된다.
- 각 Phase는 객관적인 artifact/test gate를 가지며 다음 Phase dependency가 명시된다.
