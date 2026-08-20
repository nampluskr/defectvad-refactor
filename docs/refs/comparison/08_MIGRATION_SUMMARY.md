# `defectvad`의 `cv_boilerplate` 통합을 위한 migration 요약

> **저장소 참조 안내** — 본문의 `<repo>@<commit>:<path>#<symbol>` 표기는 아래 깃허브 저장소를 가리킨다.
>
> | 표기 | 깃허브 경로 |
> |---|---|
> | `defectvad@14879ea2` | https://github.com/nampluskr/defectvad |
> | `cv_boilerplate@65d5412b` | https://github.com/nampluskr/cv_boilerplate |
> | `roi-corner-detection-ver3@8ae989a8` | https://github.com/nampluskr/roi-corner-detection-ver3 |
>
> 특정 파일은 `https://github.com/nampluskr/<repo>/blob/<commit>/<path>` 로 조회한다.


## 1. 목적과 판정 범위

이 문서는 [01~07 비교 문서](README.md)의 정적 근거를 통합해 `defectvad`의 요소를 목표 구조로 옮길 때의 판정을 요약한다. 이는 제품 구현 계획이나 source migration을 실행한 기록이 아니다. 상위 의도와 요구사항의 source of truth는 계속 `BRIEF.md`, `PRD.md`, `SPEC.md`, `PLAN.md`다.

판정 용어는 비교 README §9를 따른다.

- 재사용: 책임과 의미를 거의 바꾸지 않고 사용할 수 있다.
- 조정: adapter 또는 현재 contract에 맞춘 제한된 변경이 필요하다.
- 대체: `cv_boilerplate`의 현 기능이 같은 책임을 더 적절히 담당한다.
- 제외: 현재 목적에 불필요하거나 우연한 레거시 구조다.
- 미결정: pinned source, protocol 또는 사용자 승인이 더 필요하다.

## 2. 목표 구조 요약

```text
사용자 manifest / config / CLI override
  -> registry + validation + local asset preflight
  -> pure-PyTorch anomaly model package
  -> anomaly TaskAdapter
  -> common Trainer / checkpoint / logging
  -> metric, prediction-map artifact, visualization
  -> failure-isolated benchmark + provenance inventory
```

`확인된 사실` `cv_boilerplate`에는 registry, config validation, `TaskAdapter`, common `Trainer`, checkpoint/context, offline guard와 benchmark runner가 있다. `defectvad`에는 anomaly model, model-specific trainer, evaluator/visualizer와 dataset 구현이 있다. ROI 프로젝트에는 사용자가 model/network/head/dataset 조건을 명시해 CLI와 batch로 반복 실행하는 workflow가 있다.

근거:

- `cv_boilerplate@65d5412b:src/core/registry.py#Registry`, `src/core/adapter.py#TaskAdapter`, `src/core/engine.py#Trainer`
- `defectvad@14879ea2:src/defectvad/common/base_model.py#BaseModel`, `base_trainer.py#BaseTrainer`, `evaluator.py#Evaluator`
- `roi-corner-detection-ver3@8ae989a8:scripts/config.py#parse_args`, `scripts/batch_run.py#run`

`권고` 목표 architecture는 legacy/ROI code의 파일 구조를 복제하지 않는다. algorithm은 model package에 보존하고, lifecycle·batch·output의 차이만 adapter로 번역하며, platform 책임은 `cv_boilerplate`에 둔다.

## 3. 구성요소별 이전 판정

| 구성요소 | 기존 근거 | 판정 | 목표 위치 | 필요 검증 |
|---|---|---|---|---|
| anomalib pure-PyTorch network/loss/update | `defectvad` model package | 조정 | pinned model package | source checksum/diff와 parity |
| model-only weight save/load | `defectvad` `BaseModel` | 대체 | common checkpoint | resume 및 new-process load |
| `BaseTrainer` common loop | `defectvad` | 대체 | common `Trainer` | AMP/clip/monitor/regression |
| model-specific fit sequence | trainer subclass | 조정 | adapter hook/model method | source lifecycle parity |
| anomaly sample dict/dataset parser | legacy dataset | 조정 | dataset + anomaly adapter | split/mask/transform contract |
| `Evaluator` fixed output keys | legacy common evaluator | 대체 | adapter metric contract | image/pixel protocol |
| legacy visualizer | anomaly overlay renderer | 조정 | adapter visualization | coordinate/map/threshold diagnosis |
| module/class YAML lookup | legacy factory | 대체 | logical registry + config | unknown/duplicate validation |
| ROI model/network/head options | ROI config/factory | 조정 | model/adaptor config axes | only meaningful anomaly axes |
| ROI wrapper pre/postprocessor pattern | ROI wrapper | 조정 | anomaly adapter/model package | raw-output normalization |
| ROI CLI/batch condition visibility | ROI scripts | 재사용 | config/CLI/benchmark manifest | resolved condition parity |
| ROI factory model-name branch | `get_wrapper` | 제외 | 없음 | registry extension instead |
| ROI result CSV/failure status | Predictor/batch runner | 조정 | prediction/result artifact | sample identity/failure isolation |
| registry/config/context/checkpoint | `cv_boilerplate` core | 재사용 | common platform | current checkout reconciliation |
| offline guard/local asset error | `cv_boilerplate` core | 재사용 | common platform + asset manifest | network-blocked preflight |

## 4. 단계별 migration 방향

### 4.1 P0: 구현 기준과 protocol을 먼저 고정

`권고` 먼저 실제 `cv_boilerplate` checkout과 기존 문서의 분석 revision 차이를 대조한다. 이어 initial model/category/split/metric/compute profile, local asset, upstream source/license 및 repeated reference result를 승인한다. 이 gate가 없으면 어떤 adapter가 필요한지와 reference tolerance를 판단할 수 없다.

연결: `P0-T01`~`P0-T05`, `SC-001`, `FR-019`, `FR-020`, `FR-022`, `CON-008`~`CON-011`.

### 4.2 P1: anomaly-neutral extension point를 먼저 검증

`권고` common engine에 anomaly model명을 추가하지 않고 registry, adapter, anomaly batch/output contract, checkpoint state와 asset preflight가 표현 가능한지 검증한다. 필요 최소 fixture로 test leakage, missing asset, wrong registry name 및 checkpoint round-trip을 확인한다.

연결: `P1-T01`~`P1-T07`, `FR-005`, `FR-013`, `FR-021`, `FR-023`, `CON-005`~`CON-007`.

### 4.3 P2~P4: model은 하나씩 source audit과 reference 비교로 확장

`권고` 초기 모델은 STFPM부터 실제 source/asset/protocol 아래서 adapter를 연결한다. STFPM에서 확인되지 않은 공통 abstraction은 EfficientAD나 PatchCore에 미리 일반화하지 않는다. 두 모델 이상에서 동일하게 검증된 lifecycle requirement만 common extension point로 승격한다.

연결: `P2` STFPM, `P3` EfficientAD, `P4` PatchCore 및 inventory 확장; `FR-003`~`FR-008`, `NFR-003`, `CON-004`.

### 4.4 P5~P6: reference equivalence와 운영 안정성

`권고` batch benchmark는 모델별 run의 실패를 격리하되 final status와 leaderboard에서 실패를 숨기지 않는다. scalar metric뿐 아니라 score/map, threshold, visualization input, split, checkpoint와 provenance를 승인 tolerance로 대조한다. 이후 fast CI와 long reference profile을 분리한다.

연결: `P5`, `P6`, `FR-018`~`FR-020`, `NFR-001`, `NFR-002`, `NFR-008`, `AC-020`.

## 5. 채택하는 효과와 유지할 제약

| 채택 효과 | 유지할 제약 |
|---|---|
| 새 model의 등록/설정/실행 위치가 명시적이다. | registry가 algorithm parity 또는 license 적합성을 보장하지 않는다. |
| common engine이 AMP, clipping, checkpoint와 logging을 일관되게 제공한다. | model-specific fit/calibration을 generic engine 분기로 만들지 않는다. |
| CLI override와 manifest가 같은 resolved condition을 표현한다. | ad-hoc override 결과를 승인 benchmark와 혼합하지 않는다. |
| 실패한 batch case도 결과에서 추적 가능하다. | 실패 continuation은 silent success가 아니다. |
| local/offline policy가 자동 download를 막는다. | 필요한 asset의 source, checksum, license 및 consumer는 별도로 관리한다. |
| score/map/threshold artifact를 reference diagnosis에 사용할 수 있다. | test data로 model selection 또는 threshold calibration을 하지 않는다. |

## 6. 사용자 결정 또는 추가 근거가 필요한 항목

- 초기 승인 모델 집합, MVTec category, split, metric, repeat 수 및 compute budget
- anomalib upstream commit, license closure, reference environment와 local asset inventory
- image/pixel threshold의 source와 acceptance tolerance
- reference-only run과 product runtime의 environment 분리 방식
- inventory/benchmark manifest의 정확한 file format과 ownership
- current `cv_boilerplate@65d5412b`와 기존 문서 분석 기준 `71261cef`의 차이

이 항목을 해결하기 전에는 `08`의 판정이 구현 승인 또는 performance equivalence의 증거가 아니다.

## 7. 완료 판정

통합은 다음을 모두 만족할 때만 완료로 판단한다.

1. 각 승인 model의 source/license/local asset/lifecycle/reference protocol이 inventory에서 추적된다.
2. model algorithm은 pinned upstream과 auditable diff를 갖는다.
3. common engine과 CLI에는 anomaly task명 또는 특정 model명 분기가 없다.
4. train/evaluate/predict는 selected checkpoint를 새 process에서 재현한다.
5. approved split 및 threshold protocol에서 reference comparison이 tolerance를 통과하거나 actional diagnosis를 남긴다.
6. batch failure와 local asset failure가 성공으로 오인되지 않는다.
7. 사용자가 config, CLI와 benchmark manifest만으로 조건을 변경하고 artifact를 찾을 수 있다.

작성일: 2026-08-20  
상태: 비교 근거 기반 migration 판정 초안
