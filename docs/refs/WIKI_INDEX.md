# WIKI INDEX — Anomaly Detection Integration

> **저장소 참조 안내** — 본문의 `<repo>@<commit>:<path>#<symbol>` 표기는 아래 깃허브 저장소를 가리킨다.
>
> | 표기 | 깃허브 경로 |
> |---|---|
> | `defectvad@14879ea2` | https://github.com/nampluskr/defectvad |
> | `cv_boilerplate@65d5412b` | https://github.com/nampluskr/cv_boilerplate |
> | `roi-corner-detection-ver3@8ae989a8` | https://github.com/nampluskr/roi-corner-detection-ver3 |
>
> 특정 파일은 `https://github.com/nampluskr/<repo>/blob/<commit>/<path>` 로 조회한다.


## 1. 목적

이 문서는 `docs/refs` 문서 집합을 탐색하기 위한 단일 진입점이다.

이후 분석과 구현에서는 이 문서를 먼저 확인하고, 필요한 근거가 있을 때만 연결된 원문 문서의 관련 절 또는 식별자를 조회한다. `BRIEF.md`, `PRD.md`, `SPEC.md`, `PLAN.md`, `backlog.json` 전체를 반복해서 읽지 않는다.

## 2. 문서 참조 규칙

문서와 구현의 의미가 충돌할 때 우선순위는 다음과 같다.

```text
사용자가 명시적으로 수정한 의도
    > BRIEF.md
    > PRD.md
    > SPEC.md
    > PLAN.md
    > backlog.json
    > 현재 구현
    > 레거시 구현
```

작업별 참조 순서는 다음과 같다.

1. 이 `WIKI_INDEX.md`에서 주제와 관련 식별자를 찾는다.
2. 요구사항 판단은 `PRD.md`의 관련 `FR`, `NFR`, `CON`, `AC`, `GAP`만 조회한다.
3. 기술 설계 판단은 `SPEC.md`의 관련 절만 조회한다.
4. 구현 순서와 완료 조건은 `PLAN.md`의 관련 Phase 또는 Task만 조회한다.
5. 진행 상태와 의존성은 `backlog.json`의 관련 Task ID만 조회한다.
6. 사용자 의도나 범위 판단이 필요한 경우에만 `BRIEF.md`의 관련 절을 조회한다.

전체 원문을 다시 읽어야만 판단할 수 있다고 가정하지 않는다. 색인에 없는 내용, 문서 간 충돌, 문서 변경이 발견되면 관련 절만 확인하고 이 색인을 함께 갱신한다.

## 3. 현재 확인된 기준

- 문서 작성 기준일: 2026-08-19
- 문서가 분석한 공개 구현 기준: `nampluskr/cv_boilerplate@71261cef`
- 현재 `cv_boilerplate` 참조 checkout: `D:\_clones\cv_boilerplate@65d5412b0fa29ec817cfffc94ccfc177a4d9aad5`, `main`, clean
- 현재 `defectvad` 참조 checkout: `D:\_clones\defectvad@14879ea2a8970cee25438500e5abfeeb4be8e358`, `main`, clean
- 현재 `roi-corner-detection-ver3` 참조 checkout: `D:\_clones\roi-corner-detection-ver3@8ae989a88996441e44fb2d5296a6419a8f661220`, `main`, clean
- 알려진 계획상 충돌: `SC-001`의 checkout 부재 조건은 해소되었으나, 문서 분석 기준 `71261cef`와 현재 checkout의 차이는 아직 대조하지 않음
- 초기 권장 모델 집합: STFPM, EfficientAD, PatchCore
- 초기 권장 benchmark 범위: MVTec AD의 `bottle` category로 pipeline 고정 후 승인된 범위로 확장
- 실행 원칙: anomalib pure-PyTorch 알고리즘을 최대한 보존하고 `cv_boilerplate`가 lifecycle을 소유
- 금지 runtime: anomalib Engine, Lightning Trainer, Lightning callback lifecycle
- 성공 기준: 실행 성공만이 아니라 승인된 protocol과 tolerance를 사용한 reference 성능 재현
- asset 원칙: local asset 우선, 자동 다운로드와 silent fallback 금지
- 사용자 명시 최종 목적: `defectvad`를 `cv_boilerplate` 방식으로 통합하고 가용하다고 판정된 모든 SOTA anomaly detection 모델을 지속적으로 포함
- 사용자 운용 참고: `roi-corner-detection-ver3`의 model, network, head, dataset, CLI 및 batch orchestration

`PLAN.md`의 `P0-T01`에 따라 현재 `cv_boilerplate` checkout의 경로와 API를 `SPEC.md` 및 분석 기준 revision과 먼저 대조한다. 이 대조가 완료되기 전에는 `SC-001`을 완전히 해소된 것으로 처리하지 않는다.

## 4. 문서별 역할

| 문서 | 역할 | 주로 찾을 내용 |
|---|---|---|
| [../../../AGENTS.md](../../../AGENTS.md) | 현재 세션의 작업 지침 | 문서 전용 범위, 저장소 역할, 참조 및 검증 규칙 |
| [BRIEF.md](BRIEF.md) | Why, 방향, 범위의 source of truth | 사용자 의도, 목표와 비목표, 역할 경계, 설계 원칙 |
| [PRD.md](PRD.md) | 검증 가능한 제품 요구사항 | 기능, 품질, 제약, gap, acceptance criteria |
| [SPEC.md](SPEC.md) | 현재 구조에 맞춘 기술 설계 | contract, data flow, lifecycle, state, test, dependency |
| [PLAN.md](PLAN.md) | 의존성을 반영한 구현 순서 | P0~P6, task, gate, verification |
| [backlog.json](backlog.json) | 실행 및 진행 상태 | task ID, dependency, status, artifacts |
| [comparison/README.md](comparison/README.md) | 세 저장소 비교 문서 색인 | 비교 범위, revision, 작성 규칙과 문서 상태 |
| [comparison/01_COMPARISON_OVERVIEW.md](comparison/01_COMPARISON_OVERVIEW.md) | 구조 변화 개요 | 세 저장소 비교 초안 |
| [comparison/02_DATA_PIPELINE.md](comparison/02_DATA_PIPELINE.md) | 데이터 파이프라인 비교 | 세 저장소 비교 초안 |

`backlog.json`은 요구사항이나 설계의 source of truth가 아니다.

## 5. 주제별 빠른 탐색

| 주제 | 요구사항 | 기술 설계 | 구현 단계 |
|---|---|---|---|
| 공통 train/evaluate/predict workflow | FR-001, FR-002, FR-010, FR-011 | SPEC §2, §8, §9, §10 | P1, P2 |
| anomalib 알고리즘 보존 | FR-003, FR-004, NFR-003, CON-004 | SPEC §6, §18, §19 | P0-T04, P2-T01, P4 |
| Lightning/Engine 비의존 | CON-001~CON-003, AC-003 | SPEC §3, §6, §17 | P2, P6-T03 |
| 이질적 모델 lifecycle | FR-005, FR-006, FR-008, OOS-006 | SPEC §7, §8 | P3, P4 |
| 모델별 preprocessing | FR-007, AC-005 | SPEC §5, §6, §13 | P2, P4 |
| 공통 anomaly output | FR-013 | SPEC §4.3, §10, §12 | P1-T01, P1-T05 |
| MVTec와 split | FR-014, FR-015, CON-009, CON-010 | SPEC §5, §13 | P0-T02, P1-T04 |
| metric과 post-processing | FR-016, FR-017, AC-007 | SPEC §4.4~§4.5, §12 | P1-T05, P5 |
| checkpoint와 calibration state | FR-012, FR-021, AC-010 | SPEC §14 | P1-T03, P2-T04, P3-T04 |
| benchmark orchestration | FR-018, FR-025, AC-020 | SPEC §11, §15 | P5-T01, P5-T02, P5-T05 |
| reference equivalence | FR-019, FR-020, NFR-001, NFR-012 | SPEC §11, §17.4 | P0, P5 |
| 재현 정보와 provenance | FR-022, NFR-002, NFR-008 | SPEC §11.2, §13.4, §17 | P0, P5, P6-T04 |
| local/offline asset | FR-023, NFR-006, CON-006, CON-007 | SPEC §13.3, §16 | P0-T03, P1-T06, P6-T01 |
| 새 모델 추가 | FR-024, NFR-004, NFR-005, CON-005 | SPEC §3.3, §6, §7 | P3-T06, P4 |
| 기존 task 회귀 | NFR-013, AC-018 | SPEC §17.2 | P3-T05, P6-T02 |
| dependency와 license | NFR-014, CON-011 | SPEC §18, §19 | P0-T04, P4-T05, P6-T03 |
| 레거시 migration 판단 | OOS-004 및 BRIEF §12 | SPEC §18 | P2-T06, P4-T05 |
| 세 저장소 구조와 사용자 workflow 비교 | FR-001, FR-018, FR-024, FR-025 | [comparison/README.md](comparison/README.md) | 비교 문서 01~09 |
| dataset, transform, split, collate 비교 | FR-007, FR-014, FR-015, CON-009, CON-010 | [comparison/02_DATA_PIPELINE.md](comparison/02_DATA_PIPELINE.md) | P0-T02, P1-T04, P3-T03, P4-T01 |

## 6. BRIEF 색인

`BRIEF.md`는 사용자 의도나 범위에 관한 판단이 필요할 때만 관련 절을 조회한다.

| 절 | 내용 |
|---|---|
| §3 | 핵심 사용자 의도 |
| §4 | 프로젝트 목적 |
| §5 | anomalib에서 재사용할 것과 사용하지 않을 것 |
| §6 | `cv_boilerplate`와 anomaly integration의 책임 경계 |
| §7 | 목표 사용자 경험 |
| §8 | benchmark와 reference equivalence의 역할 |
| §9 | 프로젝트 목표 G-01~G-10 |
| §10 | 비목표 NG-01~NG-08 |
| §11 | 설계 원칙 P-01~P-10 |
| §12 | 세 레거시 프로젝트와 `cv_boilerplate`의 참고 가치 |
| §13~§16 | 초기 범위, 성공 기준, 제약, 미결정 항목 |
| §17 | 문서 chain과 각 문서의 책임 |
| §18 | 문서 우선순위와 변경 원칙 |
| §19 | 후속 분석, 레거시 해석, abstraction 추가 원칙 |
| §20~§21 | 핵심 결정 요약과 한 문장 정의 |

핵심 방향은 다음과 같다.

> Uniform user workflow, heterogeneous model internals.

레거시 코드는 요구사항, 시행착오, reference 출처를 이해하기 위한 증거이며 architecture source of truth가 아니다.

## 7. PRD 식별자 색인

### 7.1 Functional Requirements

| 범위 | 내용 |
|---|---|
| FR-001~FR-004 | 공통 workflow, anomaly task 통합, upstream 모델 재사용, 최소 adaptation |
| FR-005~FR-009 | 이질적 lifecycle, optimization, preprocessing, auxiliary stream, 학습 결과 선택 |
| FR-010~FR-013 | 평가, prediction, calibration, 공통 output semantics |
| FR-014~FR-017 | MVTec, dataset 독립성, metric, post-processing |
| FR-018~FR-020 | benchmark, reference equivalence, protocol 차이 추적 |
| FR-021~FR-025 | checkpoint, 재현 정보, local asset, 새 모델 추가, 실패 격리 |

### 7.2 Quality, Constraints, Gaps, Acceptance

| 범위 | 내용 |
|---|---|
| NFR-001~NFR-003 | reference 성능, 재현성, upstream fidelity |
| NFR-004~NFR-005 | task-agnostic engine과 확장성 |
| NFR-006~NFR-008 | offline, testability, 관찰 가능성 |
| NFR-009~NFR-014 | dataset 독립성, 유지보수성, 명시성, 수치 현실성, 회귀 방지, 최소 dependency |
| CON-001~CON-005 | pure PyTorch, runtime 책임, 알고리즘 보존, 모델명 분기 금지 |
| CON-006~CON-011 | local asset, fallback 금지, revision, leakage, split, license |
| CON-012 | 상위 문서 우선과 사용자 승인 없는 요구사항 축소 금지 |
| GAP-001~GAP-013 | 분석 기준 구현의 upstream, lifecycle, preprocessing, state, benchmark, provenance gap |
| AC-001~AC-020 | 공통 명령부터 실패 격리까지의 완료 검증 기준 |

상세 문장이 필요하면 ID 하나 또는 연속된 최소 범위만 `PRD.md`에서 조회한다.

## 8. SPEC 절 색인

| 절 | 내용 |
|---|---|
| §1 | 목적, 범위, 증거 수준 |
| §2 | 현재 architecture와 실행 lifecycle |
| §3 | 목표 architecture, dependency 방향, 최소 core extension |
| §4 | anomaly task의 batch, output, metric, post-processing contract |
| §5 | dataset integration |
| §6 | model integration과 upstream source adaptation |
| §7 | gradient, auxiliary, fitting 등 lifecycle variation |
| §8 | training flow |
| §9 | evaluation flow |
| §10 | prediction flow |
| §11 | benchmark와 reference equivalence |
| §12 | metric과 post-processing |
| §13 | configuration |
| §14 | checkpoint와 state ownership |
| §15 | validation과 error handling |
| §16 | offline 및 local asset 동작 |
| §17 | test strategy |
| §18 | 레거시 저장소에서 migration하는 원칙 |
| §19 | dependency 변경 |
| §20 | PRD traceability matrix |
| §21 | open question과 deferred decision |
| §22 | 예상 구현 영향 영역 |

설계 변경 전에는 관련 SPEC 절과 그 절이 참조하는 PRD ID를 함께 확인한다.

## 9. PLAN 및 backlog 색인

| Phase | 목표 | 대표 작업 |
|---|---|---|
| P0 | 구현 대상과 reference baseline 고정 | checkout, 범위, asset, upstream revision, baseline |
| P1 | 최소 anomaly foundation 구축 | contract, builder, checkpoint, MVTec, metric, asset preflight |
| P2 | STFPM end-to-end vertical slice 승인 | upstream import, integration, train, round-trip, acceptance |
| P3 | 이질적 lifecycle contract 안정화 | optimization spec, hooks, auxiliary loader, state, purity |
| P4 | EfficientAD와 PatchCore 통합 | 모델별 integration, shared change 심사, acceptance |
| P5 | cross-model reference equivalence 검증 | protocol diff, 실패 격리, tolerance, long run |
| P6 | offline, regression, provenance, release 검증 | offline matrix, 전체 회귀, dependency/license audit |

작업 시작 전에는 `backlog.json`에서 해당 Task ID의 `status`, `depends_on`, `scope`, `verification`, `completion_criteria`만 조회한다.

현재 계획의 첫 gate는 `P0-T01`이다. 실제 implementation checkout과 revision이 고정되지 않으면 P1 이후의 정확한 변경 파일과 regression baseline을 확정하지 않는다.

## 10. 자주 발생할 판단의 기준

### 레거시 코드에 기능이 존재하는 경우

다음을 순서대로 판단한다.

1. 현재 사용자 의도에 필요한가.
2. `cv_boilerplate`가 이미 담당하는 책임인가.
3. anomaly-specific 또는 model-specific 책임인가.
4. 과거 copy/paste 과정에서 생긴 우연한 구조인가.
5. anomalib reference 성능 재현에 필요한가.

### 새 abstraction을 추가하려는 경우

다음을 확인한다.

1. 기존 extension point로 해결할 수 있는가.
2. 최소 adapter로 해결할 수 있는가.
3. 두 개 이상의 모델에서 실제로 필요한가.
4. 공통 engine에 task명 또는 모델명 분기를 만들지 않는가.
5. reference 성능 검증과 upstream diff 검토를 어렵게 만들지 않는가.

### 문서와 코드가 충돌하는 경우

1. 확인한 사실과 revision을 기록한다.
2. 영향을 받는 `FR`, `NFR`, `CON`, `AC`를 찾는다.
3. BRIEF의 사용자 의도에 미치는 영향을 설명한다.
4. 가능한 대안을 제시한다.
5. 사용자 결정이 필요하면 상위 문서부터 수정하고 하위 문서를 동기화한다.

## 11. 색인 유지 규칙

- 상위 문서의 의도나 요구사항 ID가 변경되면 이 색인을 같은 변경에서 갱신한다.
- SPEC 절, PLAN Phase, backlog Task가 추가되거나 이름이 바뀌면 관련 표를 갱신한다.
- 확인되지 않은 내용을 이 문서에 확정 사실처럼 기록하지 않는다.
- 참조 checkout이 변경되면 확인된 revision, worktree 상태와 `SC-001` 상태를 갱신한다.
- 문서 전체 재독 대신 변경된 diff와 관련 절만 확인한다.

작성일: 2026-08-20  
상태: Navigation index, 3개 저장소 비교 범위 반영
