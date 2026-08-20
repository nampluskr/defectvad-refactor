# PLAN — Anomaly Detection Integration on `cv_boilerplate`

상위 문서: [BRIEF.md](BRIEF.md) · [PRD.md](PRD.md) · [SPEC.md](SPEC.md)

## 1. 참고 저장소

| 저장소 | 역할 |
|---|---|
| [`nampluskr/defectvad`](https://github.com/nampluskr/defectvad) | 리팩터링 대상. 20개 모델의 구현 경험과 시행착오를 참고한다 |
| [`nampluskr/cv_boilerplate`](https://github.com/nampluskr/cv_boilerplate) | 통합 기반. 순수 PyTorch 실행 프레임워크 |
| [`nampluskr/roi-corner-detection-ver3`](https://github.com/nampluskr/roi-corner-detection-ver3) | 참고. CLI·batch로 여러 모델과 실험 조건을 조립해 실행한 선례 |

### 1.1 코드 비교 분석 참조

세 저장소의 상세 비교 분석은 [`docs/refs/comparison/`](../../refs/comparison/)에 이미 작성되어 있다. 새로 분석하지 말고 아래 색인에서 필요한 문서의 해당 절만 조회한다.

| 문서 | 다루는 범위 |
|---|---|
| [README.md](../../refs/comparison/README.md) | 비교 분석의 목적, 문서 경계, 전체 색인 |
| [01_COMPARISON_OVERVIEW.md](../../refs/comparison/01_COMPARISON_OVERVIEW.md) | 전체 실행 흐름, 사용자 운용 구조와 상위 구조의 대응 |
| [02_DATA_PIPELINE.md](../../refs/comparison/02_DATA_PIPELINE.md) | dataset, transform, split, collate, dataloader |
| [03_MODEL_AND_ADAPTER.md](../../refs/comparison/03_MODEL_AND_ADAPTER.md) | model, wrapper, adapter, loss와 optimizer 경계 |
| [04_EXECUTION_LIFECYCLE.md](../../refs/comparison/04_EXECUTION_LIFECYCLE.md) | trainer, evaluator, predictor의 train/evaluate/predict 흐름 |
| [05_OUTPUT_AND_VISUALIZATION.md](../../refs/comparison/05_OUTPUT_AND_VISUALIZATION.md) | output, metric, post-processing, threshold, visualizer |
| [06_CLI_AND_BATCH_ORCHESTRATION.md](../../refs/comparison/06_CLI_AND_BATCH_ORCHESTRATION.md) | CLI, notebook 조립 경로, 반복 실행과 benchmark orchestration |
| [07_PLATFORM_MECHANISMS.md](../../refs/comparison/07_PLATFORM_MECHANISMS.md) | registry, factory/builder, config, checkpoint, logging, offline |
| [08_MIGRATION_SUMMARY.md](../../refs/comparison/08_MIGRATION_SUMMARY.md) | 변경·일반화 요약, 효과, 제약, gap과 이전 판정 |
| [09_EVIDENCE_INDEX.md](../../refs/comparison/09_EVIDENCE_INDEX.md) | 주요 판단과 코드 근거의 역방향 색인 |

Phase별 주요 참조:

- **P1 공통 구조 설계** — 02(데이터), 03(adapter 경계), 07(registry·config)
- **P2·P4 모델 통합** — 03(wrapper/adapter), 04(lifecycle)
- **P3·P5 성능 검증** — 05(metric·threshold), 08(gap 판정)
- **특정 결론의 코드 근거 추적** — 09

`docs/refs/`의 문서는 이전 분석 결과로 보존하며 수정하지 않는다. 필요한 부분만 참조한다.

## 2. 버전 전략

30개 모델을 한 번에 구현하지 않는다. 이상탐지 패러다임별로 나누어 단계적으로 확장한다.

| 버전 | 모델 | 데이터셋 | 목적 |
|---|---|---|---|
| **v0.1** (이번 단계) | STFPM, EfficientAD | MVTec AD | 필수 핵심 최소 모델로 전체 구조를 확립하고 reference 성능 재현을 검증한다 |
| v0.2 | (v0.1과 동일) | + BTAD, VisA | 모델을 늘리지 않고 데이터셋만 확장해 데이터셋 추상화를 검증한다 |
| v0.3 | defectvad 나머지 18개 모델 (패러다임별 순차 확장) | (v0.2와 동일) | 기존 구현 경험이 있는 모델로 범위를 한정해 확장한다 |
| v0.4 | anomalib 나머지 10개 모델 | (v0.2와 동일) | 기존에 다루지 않았던 모델까지 확대한다 |

v0.1에서 두 모델을 고른 이유는 lifecycle 특성이 서로 다르기 때문이다. STFPM은 단순 gradient training(SGD, 스케줄러 없음)이고, EfficientAD는 auxiliary 데이터·teacher 통계·검증 전 분위수 계산이 필요하다. 이 두 축을 모두 수용하면 boilerplate의 확장 지점이 검증된 것으로 본다.

모델 확장과 데이터셋 확장은 서로 독립적인 축이므로 버전을 나누어 진행한다. v0.2에서 모델을 고정한 채 데이터셋만 늘리면, 이후 모델이 늘어난 상태에서 데이터셋 문제가 겹쳐 원인을 가리기 어려워지는 상황을 피할 수 있다. 모든 확장은 v0.1에서 확립된 구조를 **그대로 사용**해야 하며, 확장 과정에서 공통 구조를 다시 설계해야 한다면 v0.1의 추상화가 부족했다는 신호로 본다.

## 3. v0.1 단계

| Phase | 단계 | 목적 | 완료 조건 |
|---|---|---|---|
| **P0** | 기준 확정 | 대상 anomalib commit, boilerplate 구조, 로컬 자산 경로를 고정한다 | 복사 대상 파일 목록과 reference 성능 기준값이 문서화됨 |
| **P1** | 공통 구조 설계 | anomaly task의 데이터셋·metric·wrapper 경계를 정의한다 | MVTec 로딩과 AUROC 산출이 독립적으로 검증되고, 데이터셋 인터페이스가 MVTec 구조에 결합되지 않음 |
| **P2** | STFPM 통합 | 첫 모델로 end-to-end 흐름을 완성한다 | train/evaluate/predict가 동작하고 결과가 산출됨 |
| **P3** | STFPM 성능 검증 | anomalib reference 성능을 재현한다 | 대표 3개 카테고리에서 사용자 실행 결과가 reference와 비교 가능 (PRD §5) |
| **P4** | EfficientAD 통합 | 다른 lifecycle을 공통 구조로 수용한다 | auxiliary 데이터·통계 계산이 wrapper에서 처리됨 |
| **P5** | EfficientAD 성능 검증 | 두 번째 모델의 reference 성능을 재현한다 | 대표 3개 카테고리에서 사용자 실행 결과가 reference와 비교 가능 (PRD §5) |
| **P6** | 구조 정리 및 문서화 | 모델 추가 절차를 반복 가능한 형태로 확정한다 | 이후 버전에서 새 모델·데이터셋을 추가하는 절차가 문서화됨 |

각 Phase는 이전 Phase에 의존하며, P2~P3과 P4~P5는 각각 하나의 모델에 대한 통합·검증 쌍이다.

성능 검증 Phase(P3, P5)는 에이전트가 임의로 학습을 실행하지 않는다. 구현 완료 후 실행 명령어를 제시하고, 사용자가 터미널에서 실행한 결과를 피드백받아 판정한다([PRD.md](PRD.md) §5).

## 4. v0.2 이후 확장 방향

v0.1에서 확정된 절차와 구조를 그대로 사용해 확장한다. 각 버전의 구체적인 범위와 순서는 이전 버전 완료 후 결정한다.

### 4.1 v0.2 — 데이터셋 확장

모델은 v0.1의 두 개로 고정한 채 데이터셋만 늘린다.

| 데이터셋 | 로컬 경로 | 특성 |
|---|---|---|
| BTAD | `/mnt/d/datasets/btad` | 3개 카테고리, MVTec과 다른 디렉터리 구조 |
| VisA | `/mnt/d/datasets/visa` | 12개 카테고리, split 정의 방식이 다름 |

목적은 모델·metric 로직이 특정 데이터셋 구조에 결합되지 않았음을 확인하는 것이다. 새 데이터셋을 추가할 때 모델 코드나 wrapper를 수정해야 한다면 데이터셋 추상화가 잘못된 것이다.

### 4.2 v0.3 — 모델 확장 (패러다임별 순서)

defectvad에서 이미 다룬 20개 모델 중 v0.1에서 구현한 2개를 제외한 나머지를 이상탐지 패러다임별로 묶어 순차 확장한다. 같은 패러다임의 모델은 lifecycle이 유사하므로, 그룹의 첫 모델에서 확장 지점이 확인되면 나머지는 반복 작업이 된다.

| 순서 | 패러다임 | 모델 | lifecycle 특성 |
|---|---|---|---|
| 1 | Knowledge Distillation | reversedistill, fre | teacher/student feature 비교 (STFPM·EfficientAD와 동일 계열) |
| 2 | Memory Bank / Statistics | patchcore, padim, dfkde, dfm | gradient training 없음, fitting 후 bank/통계 구축 |
| 3 | Normalizing Flow | fastflow, cflow, csflow, uflow | density estimation 기반 |
| 4 | Reconstruction | draem, dsr, ganomaly | 재구성 오차 기반, 합성 이상 생성 포함 |
| 5 | Feature Adaptation | cfa, supersimplenet, uninet | feature 공간 적응·학습 |
| 6 | Foundation Model | anomalydino, dinomaly | 사전학습 대형 backbone 활용 |

각 모델은 v0.1과 동일하게 통합 → reference 성능 검증 순으로 진행한다. 패러다임 순서는 lifecycle 난이도와 v0.1 구조와의 거리를 기준으로 정했으며, 2번 그룹(gradient training이 없는 모델)이 boilerplate 추상화의 가장 큰 시험대다.

### 4.3 v0.4 — 나머지 모델

anomalib에는 있으나 defectvad에서 다루지 않은 10개 모델이다. v0.3 완료 후 필요에 따라 선택한다.

`anomalyvfm`, `cfm`, `general_ad`, `glass`, `inp_former`, `l2bt`, `patchflow`, `super_add`, `vlm_ad`, `winclip`

## 5. 제약

모든 Phase는 [BRIEF.md](BRIEF.md)의 세 가지 원칙을 따른다.

- 원칙 1 — anomalib 모델 코드는 경로를 제외하고 수정하지 않는다.
- 원칙 2 — Lightning을 사용하지 않으며, boilerplate는 필요에 따라 개선한다.
- 원칙 3 — 로컬 자산만 사용하며, 자동 다운로드 없이 오프라인으로 실행한다.

---

작성일: 2026-08-20
문서 상태: Initial Plan
