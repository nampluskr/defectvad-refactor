# SPEC — Anomaly Detection Integration on `cv_boilerplate`

> **저장소 참조 안내** — 본문의 `<repo>@<commit>:<path>#<symbol>` 표기는 아래 깃허브 저장소를 가리킨다.
>
> | 표기 | 깃허브 경로 |
> |---|---|
> | `defectvad@14879ea2` | https://github.com/nampluskr/defectvad |
> | `cv_boilerplate@65d5412b` | https://github.com/nampluskr/cv_boilerplate |
> | `roi-corner-detection-ver3@8ae989a8` | https://github.com/nampluskr/roi-corner-detection-ver3 |
>
> 특정 파일은 `https://github.com/nampluskr/<repo>/blob/<commit>/<path>` 로 조회한다.


문서 상태: Initial Technical Specification  
상위 문서: `BRIEF.md`, `PRD.md`  
작성일: 2026-08-19  
현재 구현 분석 기준: `nampluskr/cv_boilerplate@71261cef`

## 1. Purpose and Scope

### 1.1 목적

이 문서는 `BRIEF.md`의 사용자 의도와 `PRD.md`의 검증 가능한 요구사항을 현재 `cv_boilerplate`
architecture에서 구현하기 위한 기술 설계를 정의한다.

핵심 목표는 다음과 같다.

- 기존 `train`, `evaluate`, `predict`, `benchmark` entry point를 유지한다.
- anomalib의 pure-PyTorch algorithm code를 가능한 한 그대로 유지한다.
- anomalib Lightning wrapper와 Engine이 담당하던 model-specific lifecycle만 `cv_boilerplate`의 기존
  adapter/hook 경계로 옮긴다.
- gradient training, auxiliary data, statistics fitting, memory bank 및 no-gradient fitting을 공통 engine의
  model-name 분기 없이 수용한다.
- reference protocol과 결과 차이를 재현 가능하게 기록한다.

Implements: `FR-001`~`FR-025`, `NFR-001`~`NFR-014`, `CON-001`~`CON-012`

### 1.2 설계 기준과 증거 수준

- `[확인]`은 실제 source에서 확인한 현재 동작이다.
- `[설계]`은 이 SPEC이 요구하는 target 동작이다.
- `[미확정]`은 구현 전에 사용자 또는 reference baseline으로 결정해야 하는 사항이다.

현재 작업 directory에는 `cv_boilerplate` source가 없으므로, 현재 architecture에 대한 사실은 PRD와 동일하게
공개 revision `71261cef`의 분석 결과를 기준으로 한다. 분석에 사용한 anomalib `c9eeefff`는 reference 구조를
파악하기 위한 기준일 뿐 target revision으로 확정하지 않는다.

### 1.3 포함 범위

- 현재 execution/config/construction/checkpoint/metric/output 흐름의 최소 확장
- Anomaly Detection task contract
- MVTec AD dataset contract
- anomalib pure model과 project-specific integration code의 경계
- model lifecycle variation
- train/evaluate/predict/benchmark call flow
- metric, post-processing, checkpoint 및 offline asset contract
- reference equivalence 기록과 검증
- legacy migration 판단과 dependency 영향

### 1.4 다루지 않는 범위

- 구현 순서, Phase, backlog 및 일정
- 초기 모델과 category의 최종 목록
- 임의의 metric tolerance 수치
- anomalib 전체 runtime 또는 API 호환 계층
- legacy framework의 재구성
- 모든 anomalib 모델과 dataset의 즉시 지원

Guards: `OOS-001`~`OOS-011`

## 2. Current Architecture Summary

### 2.1 Execution entry point

`[확인]` `python -m src`의 `src/__main__.py`가 network guard를 먼저 활성화하고 task package를 import해
registry를 채운 뒤 CLI command를 dispatch한다. 사용자-facing lifecycle은 `src/cli/commands.py`의
`train`, `evaluate`, `predict`, `run_benchmark`가 소유한다.

Implements: `FR-001`, `CON-003`

### 2.2 Config loading과 object construction

`[확인]` 현재 흐름은 다음과 같다.

```text
YAML path
  -> load_and_merge_base
  -> dotted --set override
  -> validate_config
  -> build_transforms
  -> MODELS / LOSSES / METRICS / ADAPTERS registry build
  -> DATASETS registry build
  -> build_dataloader
  -> build_optimizer / build_scheduler
```

`[확인]` 동일한 component construction 함수가 CLI와 benchmark runner에 중복되어 있다. 이 SPEC은 새 config
system을 만들지 않으며, 이 중복을 확장 과정에서 단일 construction path로 정리하는 것을 허용한다. 정리 자체는
anomaly 전용 abstraction이 아니라 현재 동작을 보존하는 공통화여야 한다.

Implements: `FR-002`, `FR-024`, `NFR-010`, `CON-003`

### 2.3 Task, model, dataset 관계

`[확인]` 별도 Task object는 없다. Task package가 import 시 Dataset, Transform, Model, Loss, Metric, Adapter를
각 registry에 등록한다. `TaskAdapter`가 batch forwarding, loss, metric, prediction, collate 및 lifecycle hook을
담당한다.

`[설계]` Anomaly integration도 이 구조를 유지한다. 새 Task class나 별도 anomaly Engine을 추가하지 않는다.

Implements: `FR-002`, `NFR-004`, `CON-005`

### 2.4 Training lifecycle

`[확인]` `Trainer.fit`은 다음 순서를 사용한다.

```text
model.to(device)
adapter.on_fit_start
for epoch:
    adapter.on_epoch_start
    Trainer._train_epoch
        model.train
        every batch: zero_grad -> adapter.train_step -> backward -> optimizer.step
    scheduler.step
    Trainer.evaluate(valid)
    adapter.on_epoch_end
    best/last checkpoint save
best model reload
adapter.on_fit_end
finalized model_state를 best checkpoint에 재저장
```

`[확인]` optimizer와 differentiable loss가 항상 존재한다고 가정하므로 PatchCore/PaDiM 계열을 정상적으로
표현하지 못한다. validation 직전 hook도 없어 EfficientAD의 validation map quantile lifecycle과 일치하지 않는다.

Implements: `FR-005`, `FR-006`, `GAP-002`, `GAP-003`, `GAP-006`

### 2.5 Evaluation lifecycle

`[확인]` `evaluate`는 config로 model과 adapter를 새로 만들고 checkpoint의 model state만 복원한 뒤
`Trainer.evaluate`로 metric을 계산한다. Adapter state는 checkpoint 대상이 아니다.

`[확인]` `Trainer.evaluate`는 `model.eval`, `adapter.reset_metrics`, `torch.no_grad`, batch별 `eval_step`과
metric update, 최종 compute를 수행한다.

Implements: `FR-010`, `FR-012`, `GAP-007`

### 2.6 Prediction lifecycle

`[확인]` `predict`는 단일 파일 또는 directory의 PIL image를 eval transform에 통과시키고 adapter collate 후
`predict_step`을 호출한다. 결과는 `predictions/predict.json`과 task-specific artifact로 저장한다.

`[확인]` 현재 anomaly threshold는 새 adapter에 복원되지 않아 calibrated label이 `None`이 될 수 있다.

Implements: `FR-011`, `FR-012`, `FR-013`, `GAP-007`

### 2.7 Benchmark lifecycle

`[확인]` benchmark runner는 base config와 split override를 merge하고 control field를 검사한 뒤 split마다
train/valid/test/profile을 실행한다. 한 split의 exception은 failed row로 격리되고 나머지 split은 계속된다.
결과는 control report, resolved config, metrics, environment 및 leaderboard에 기록된다.

`[확인]` 현재 anomaly benchmark는 동일 조건 모델 비교용 smoke benchmark이며, 모델별 anomalib reference와의
equivalence pair를 표현하지 않는다.

Implements: `FR-018`, `FR-019`, `FR-025`, `GAP-010`

### 2.8 Checkpoint와 state ownership

`[확인]` checkpoint는 model, optimizer, scheduler, scaler, epoch, best metric, monitor, config, environment 및
RNG state를 보존한다. Adapter/post-processor state는 보존하지 않는다.

`[설계]` 기존 checkpoint container를 유지하고 adapter state와 global step, protocol identity를 확장 필드로
추가한다. 별도 anomaly checkpoint format을 만들지 않는다.

Implements: `FR-021`, `FR-022`, `NFR-002`, `GAP-007`, `GAP-011`

### 2.9 Metrics와 output

`[확인]` 현재 anomaly adapter는 image/pixel `BinaryAUROC`를 update하고 gaussian smoothing 및 valid-only F1
threshold를 계산한다. Threshold는 model state가 아니라 adapter field다.

`[확인]` output 관리는 resolved YAML, JSON, CSV, log, checkpoint, visualization 및 leaderboard로 이미
구성되어 있다. 이 구조를 확장하고 별도 experiment tracking system을 도입하지 않는다.

Implements: `FR-016`, `FR-017`, `FR-022`, `NFR-008`, `OOS-009`

## 3. Target Architecture

### 3.1 Responsibility boundary

`[설계]` 책임 경계는 다음과 같다.

```text
cv_boilerplate core
  CLI / config / construction / device / common Trainer
  checkpoint container / run outputs / benchmark orchestration
  generic optimization execution / generic hook invocation
                    |
                    v
Anomaly task integration
  sample and output contract / collate / metric routing
  common anomaly post-processing / visualization
  adapter state serialization
                    |
                    v
Per-model integration
  upstream model invocation / loss invocation / trainable parameters
  optimizer and scheduler specification / auxiliary loader consumption
  model-specific prepare, validation-prepare, finalize
  upstream output conversion
                    |
                    v
Vendored or pinned anomalib pure-PyTorch algorithm
  architecture / loss / anomaly map / memory bank / statistics algorithm

Dataset integration
  directory parsing / sample metadata / split realization / mask loading
  model-independent geometric transform of image and mask
```

### 3.2 Dependency direction

`[설계]` dependency는 core에서 구체 모델로 향하지 않는다.

- core는 `TaskAdapter` contract와 generic optimization/state contract만 안다.
- anomaly common code는 upstream concrete model을 import하지 않는다.
- per-model integration은 anomaly common contract와 선택된 upstream model을 안다.
- upstream model code는 `cv_boilerplate` core, CLI, Dataset 또는 benchmark를 import하지 않는다.
- Dataset은 model 이름, optimizer, metric implementation을 알지 않는다.

Implements: `FR-015`, `FR-024`, `NFR-004`, `NFR-005`, `NFR-009`, `NFR-010`, `CON-005`

### 3.3 최소 core extension

`[설계]` 기존 `TaskAdapter`를 유지하면서 다음 capability만 추가한다.

1. optimizer가 없을 수 있는 train step
2. adapter가 trainable parameter와 optimizer/scheduler cadence를 정의하는 optimization specification
3. validation 직전과 직후의 optional hook
4. named auxiliary loader를 기존 loader mapping에 포함하는 construction
5. adapter state의 checkpoint save/load
6. epoch 및 step scheduler cadence의 generic 실행

별도 anomaly Trainer, callback framework, event bus 또는 general-purpose workflow engine은 추가하지 않는다.

Implements: `FR-005`, `FR-006`, `FR-008`, `FR-021`, `NFR-010`, `CON-001`, `OOS-004`, `OOS-006`

## 4. Anomaly Detection Task Contract

### 4.1 기존 extension 사용

`[설계]` Anomaly Detection은 기존 Dataset, Transform, Model, Metric, Adapter registry와 `TaskAdapter`를 사용한다.
별도 Task class는 만들지 않는다.

Implements: `FR-001`, `FR-002`, `FR-024`, `NFR-004`

### 4.2 Primary batch contract

`[설계]` primary batch는 현재 convention을 유지한다.

```text
images: Tensor[B, 3, H, W], float
targets: list[dict]
```

Train target는 normal-only one-class model에서 빈 dict일 수 있다. Evaluation target는 다음 공통 field를 사용한다.

```text
label: scalar integer tensor, 0=normal, 1=anomalous
mask: Tensor[H, W], integer binary mask
sample_id: stable dataset-independent identifier
path: source image path or equivalent source identifier
metadata: optional dataset-specific information
```

`sample_id`, `path`, `metadata`의 실제 전달 위치는 현재 tuple convention과 호환되도록 구현 시 확정하되,
model/loss/metric이 dataset-specific metadata에 의존해서는 안 된다.

Implements: `FR-013`, `FR-014`, `FR-015`, `NFR-009`

### 4.3 Model output contract

`[설계]` adapter가 core에 노출하는 normalized output은 다음 의미를 가진다.

```text
pred_score: Tensor[B]
anomaly_map: Tensor[B, H, W] | None
extras: optional mapping not consumed by common metric code
```

- `pred_score`는 클수록 anomalous해야 한다.
- localization 모델의 `anomaly_map`은 evaluation mask와 공간적으로 대응해야 한다.
- upstream의 `InferenceBatch`, dict, `(B,1,H,W)` 등 차이는 per-model integration이 변환한다.
- conversion은 값의 의미를 바꾸지 않으며 squeeze, field mapping 및 명시된 interpolation만 허용한다.

Implements: `FR-011`, `FR-013`, `FR-016`, `FR-017`

### 4.4 Metric input boundary

`[설계]` anomaly common adapter는 normalized output과 common target만 metric에 전달한다. Metric은 model 또는
dataset directory를 참조하지 않는다.

- image metric: `pred_score`, `label`
- pixel metric: `anomaly_map`, `mask`
- threshold metric: calibrated score/map와 binary target
- region metric: anomaly map과 connected-component semantics가 보존된 mask

Implements: `FR-015`, `FR-016`, `NFR-009`

### 4.5 Post-processing boundary

`[설계]` 처리 순서를 다음처럼 분리한다.

```text
upstream raw output
  -> per-model algorithm-defined map processing
  -> common representation conversion
  -> protocol-defined normalization/smoothing
  -> threshold application
  -> metric or user-facing prediction
```

Algorithm 의미에 포함된 처리는 upstream/per-model integration에 남기고, validation-derived threshold와 공통
prediction 변환은 stateful anomaly post-processing에 둔다.

Implements: `FR-003`, `FR-004`, `FR-017`, `NFR-003`, `CON-004`

## 5. Dataset Integration

### 5.1 Dataset 책임

`[설계]` Dataset은 directory parsing, sample identity, image/mask loading, label 생성 및 split membership만
소유한다. Model-specific normalization, optimizer, statistics fitting 및 threshold는 소유하지 않는다.

Implements: `FR-014`, `FR-015`, `NFR-009`, `NFR-010`

### 5.2 MVTec AD

`[설계]` 기존 `MVTecAnomaly`의 다음 동작을 유지한다.

- category root의 `train/good`, `test/<defect_type>`, `ground_truth/<defect_type>` parsing
- train은 normal sample만 반환
- `good` evaluation image는 label 0과 all-zero mask
- abnormal image는 label 1과 대응 ground-truth mask
- mask를 `{0,1}` integer로 변환
- image와 mask의 geometric transform alignment 유지
- stable split file 사용 및 train/valid/test disjoint 검증 가능

`[미확정]` Reference-equivalence run에서 disjoint valid/test를 사용할지 anomalib reference와 같은 split을 사용할지는
결정되지 않았다. Dataset 구현은 두 정책을 모두 explicit split manifest로 표현할 수 있어야 한다.

Implements: `FR-009`, `FR-014`, `CON-009`, `CON-010`

### 5.3 Split manifest

`[설계]` 모든 benchmark split은 materialized sample identifier 또는 동일하게 재현 가능한 split specification으로
기록한다. Manifest에는 dataset identity, category, seed, source population 및 train/valid/test membership을
포함한다. Validation과 test가 같으면 이를 명시적인 protocol flag로 기록한다.

Implements: `FR-009`, `FR-020`, `FR-022`, `NFR-002`, `CON-009`, `CON-010`

### 5.4 Transform ownership

`[설계]` Dataset-independent geometric transform은 existing transform registry를 사용한다. Model-specific config가
reference transform을 선택하며, 모든 anomaly model에 하나의 normalization을 강제하지 않는다.

- image와 mask에 같은 resize/crop geometry 적용
- mask interpolation은 nearest
- image normalization은 model config가 선택
- EfficientAD처럼 raw `[0,1]` input을 요구하는 모델은 Dataset이 아니라 model config가 normalization을 끈다.
- predict와 evaluation은 같은 resolved eval transform을 사용한다.

Implements: `FR-007`, `FR-014`, `FR-015`, `GAP-004`

### 5.5 Auxiliary dataset

`[설계]` primary `data` 구조를 유지하면서 model config가 named auxiliary data specification을 선언할 수 있게
확장한다. Construction은 기존 Dataset/Transform/DataLoader registry를 재사용해 loader mapping에 이름으로
추가한다. Adapter는 선언된 이름만 소비한다.

예시 의미:

```yaml
data:
  auxiliary:
    imagenette:
      name: local_image_folder
      root: /local/path
      batch_size: 1
      transform: {name: efficientad_penalty, params: {}}
```

이 예시는 schema 방향이며 최종 key 이름은 구현 전 config compatibility 검토로 확정한다.

Implements: `FR-008`, `FR-023`, `NFR-006`, `CON-006`

### 5.6 Future dataset

`[설계]` VisA/BTAD는 초기 구현 대상으로 확정하지 않는다. 향후 각 Dataset이 common sample contract로 parsing하고
동일 adapter/metric/model integration을 재사용할 수 있어야 한다.

Implements: `FR-015`, `NFR-009`, `OOS-008`

## 6. Model Integration

### 6.1 Upstream code reuse policy

`[설계]` 통합 대상마다 target anomalib revision과 pure-PyTorch dependency closure를 먼저 확정한다.

- algorithm file은 upstream naming과 module separation을 가능한 한 유지한다.
- formatting, output type 및 import convenience를 이유로 algorithm을 rewrite하지 않는다.
- project-specific code는 upstream file 밖에서 loss invocation, output conversion, lifecycle 및 asset injection을
  담당한다.
- 불가피한 upstream 수정은 source manifest와 diff record에 이유, 영향, 검증을 기록한다.

Implements: `FR-003`, `FR-004`, `NFR-003`, `CON-004`, `CON-008`, `CON-011`

### 6.2 Source transport boundary

`[미확정]` Vendoring과 별도 sync 방식은 확정하지 않는다. 어느 방식을 선택해도 다음 artifact가 필요하다.

- upstream repository URL
- version/commit
- copied source path 목록
- license/notice
- local patch 또는 diff
- source checksum
- dependency list

제품 runtime은 anomalib Lightning/Engine을 import하지 않는다.

Implements: `FR-003`, `FR-022`, `NFR-003`, `CON-002`, `CON-008`, `CON-011`

### 6.3 Per-model adapter

`[설계]` 기존 `TaskAdapter`가 model-specific integration point다. Anomaly common adapter는 output/metric/prediction
공통 동작을 제공하고, 실제 차이가 있는 모델은 이를 확장한 per-model adapter 또는 동등한 구성 객체로 다음을
정의한다.

- upstream training forward와 loss 호출
- trainable parameter selection
- optimizer/scheduler specification
- auxiliary loader 사용
- fit/validation/finalize hook
- upstream output conversion
- adapter/post-processing state

이 계층은 Trainer, Dataset parsing, logging, checkpoint file I/O 또는 benchmark orchestration을 소유하지 않는다.
따라서 legacy trainer나 LightningModule을 이름만 바꿔 복제하는 구조가 아니다.

Implements: `FR-004`, `FR-005`, `FR-006`, `FR-008`, `FR-024`, `NFR-010`, `OOS-004`

### 6.4 Model construction과 pretrained weight

`[설계]` Model registry는 upstream pure model 또는 이를 포함하는 최소 `nn.Module` container를 생성한다.
Pretrained architecture는 network-enabled constructor를 사용하지 않고 architecture와 local state loading을
분리한다.

- required weight path는 resolved config에 존재해야 한다.
- expected source, checksum, state-dict key contract를 검증한다.
- teacher/student처럼 일부 submodule만 load할 때 target submodule을 명시한다.
- required weight는 strict load를 기본으로 한다.
- missing/mismatch는 `LocalAssetError` 계열의 user-facing error다.

Implements: `FR-023`, `NFR-006`, `NFR-011`, `CON-006`, `CON-007`

### 6.5 Train/eval mode

`[설계]` Upstream model이 train/eval mode에 따라 output을 바꾸는 동작은 보존한다. Frozen teacher는 outer
`model.train()` 이후에도 eval 상태와 `requires_grad=False`를 유지해야 한다. 이 불변 조건은 pure model 또는
minimal container의 state behavior로 보장하고 test한다.

Implements: `FR-003`, `FR-005`, `NFR-003`, `CON-004`

### 6.6 Loss handling

`[설계]` Loss가 upstream 별도 module이면 그대로 생성해 per-model adapter가 호출한다. Loss가 model forward에
내장되어 있으면 adapter는 반환된 component를 합성하되 reference 식을 변경하지 않는다. Common anomaly
adapter는 서로 다른 모델의 valid loss를 공통 의미로 간주하지 않는다.

Implements: `FR-003`, `FR-006`, `NFR-003`

### 6.7 Optimization specification

`[설계]` `TaskAdapter`는 model과 resolved optimizer config를 바탕으로 다음 generic 정보를 core에 제공한다.

```text
optimizer: torch optimizer or None
scheduler: torch scheduler or None
scheduler_interval: "step" or "epoch"
trainable parameter groups
gradient clipping policy
```

구체 class 이름은 구현 시 기존 builder와의 compatibility를 고려해 정한다. Core는 interval과 optimizer 존재
여부만 처리하며 model 이름을 검사하지 않는다.

Implements: `FR-006`, `NFR-004`, `CON-005`, `GAP-003`

### 6.8 Model-specific state

`[설계]` 다음 상태는 가능한 경우 `nn.Module` parameter/buffer로 보존한다.

- teacher mean/std
- map quantile
- feature statistics
- memory bank/coreset
- model-specific normalization constant

Task-level threshold, score range 및 output calibration처럼 algorithm model 밖의 상태는 adapter state에 둔다.

Implements: `FR-012`, `FR-017`, `FR-021`

## 7. Model Lifecycle Variations

### 7.1 공통 lifecycle capability

`[설계]` Engine은 다음 generic 시점만 호출한다.

```text
fit preparation
train epoch start
train step
validation preparation
evaluation step
validation completion
fit finalization
```

현재 hook을 재사용하고 validation preparation/completion만 최소 추가한다. Hook invocation은 모든 task에 동일하며
model-name 조건을 갖지 않는다.

Implements: `FR-005`, `NFR-004`, `NFR-005`, `CON-005`

### 7.2 Lifecycle 비교

| 유형 / 예시 | Preparation | Training | Validation | Finalize | Inference | Checkpoint state | Required extension |
|---|---|---|---|---|---|---|---|
| Standard gradient | asset 확인 | loss, backward, optimizer step | metric | 선택적 calibration | forward | model/optimizer/scheduler | 기존 loop |
| STFPM teacher/student | teacher local weight, freeze/eval | student-only SGD, upstream loss | upstream anomaly map | threshold/post-process calibration | teacher/student discrepancy | teacher/student, calibration | parameter selection, adapter state |
| EfficientAD teacher/student/AE | teacher local weight, Imagenette loader, train-set mean/std | student+AE loss와 auxiliary penalty | validation 전에 normal map quantile 계산 | threshold/post-process calibration | normalized ST/AE map 결합 | model mean/std/quantiles, calibration | named auxiliary loader, validation hook, cadence |
| Feature statistics / PaDiM | backbone local weight | no-grad feature collection | validation 전에 statistics fit 완료 | 필요 시 threshold | Mahalanobis map | fitted statistics, selected feature indices | optimizer None, validation preparation |
| Memory bank / PatchCore | backbone local weight | no-grad embedding collection | validation 전에 coreset/memory finalize | 필요 시 threshold | nearest-neighbor score/map | memory bank, coreset state | optimizer None, validation preparation |
| No-gradient/post-fit | assets 확인 | state collection 또는 없음 | finalized state로 평가 | state completion | model-specific | finalized state | optional loss, optional optimizer |

`[미확정]` STFPM/EfficientAD/PatchCore/PaDiM 중 어떤 조합을 최초 acceptance model로 사용할지는 정하지 않는다.

Implements: `FR-005`, `FR-006`, `FR-008`, `AC-004`, `OOS-006`, `OOS-007`

### 7.3 No-optimizer train step

`[설계]` `adapter.train_step` 결과의 loss는 optimizer가 없는 lifecycle에서 `None`일 수 있다. Engine behavior는
다음과 같다.

- optimizer가 있으면 differentiable scalar loss를 요구하고 기존 AMP/backward/clip/step을 수행한다.
- optimizer가 없으면 `train_step`의 state collection side effect만 허용하고 backward/step을 수행하지 않는다.
- loss가 필요한 mode에서 누락되거나 optimizer 없는 mode에서 gradient update를 요구하면 validation error다.

Implements: `FR-005`, `FR-006`, `GAP-002`, `GAP-006`

### 7.4 Validation preparation

`[설계]` `Trainer.evaluate(..., split="valid")` 직전에 adapter validation-preparation hook을 호출한다.

- EfficientAD는 current epoch model로 validation normal map quantile을 계산한다.
- PatchCore/PaDiM은 수집된 train state를 inference-ready state로 finalize한다.
- Hook은 전달받은 loader mapping에서 명시적으로 허용된 train/valid/auxiliary data만 사용한다.
- final test evaluate에서는 calibration/fitting hook을 자동 재실행하지 않고 checkpoint state를 사용한다.

Implements: `FR-005`, `FR-009`, `FR-012`, `CON-009`

### 7.5 Lifecycle state transition

`[설계]` Model/adapter state는 최소한 `unprepared`, `collecting/training`, `inference_ready` 상태를 구분할 수 있어야
한다. 구현이 enum을 필요로 하는지는 모델 통합에서 결정하되, inference-ready가 아닌 checkpoint로 evaluate나
predict하면 명확히 실패해야 한다.

Implements: `FR-010`, `FR-011`, `FR-021`, `NFR-011`

## 8. Training Flow

### 8.1 Construction과 fit call flow

`[설계]` Target call flow는 다음과 같다.

```text
CLI train
  -> resolve_config / validate_config
  -> apply_network_policy
  -> create run_dir and save resolved config
  -> RunContext seed/device setup
  -> build primary and named auxiliary transforms
  -> build model, loss, metrics, adapter
  -> build primary and named auxiliary datasets/loaders
  -> adapter creates generic optimization specification
  -> Trainer.fit
       -> adapter.on_fit_start(model, loader_mapping, device)
       -> epoch loop
            -> adapter.on_epoch_start
            -> generic train epoch
                 -> adapter.train_step
                 -> optional optimization based on specification
                 -> optional step scheduler
            -> optional epoch scheduler
            -> adapter.on_validation_start
            -> Trainer.evaluate(valid)
            -> adapter.on_validation_end
            -> adapter.on_epoch_end
            -> selection checkpoint
       -> selected model reload when selection applies
       -> adapter.on_fit_end
       -> final model + adapter state checkpoint
  -> final metrics, protocol and environment output
```

Implements: `FR-001`, `FR-005`, `FR-006`, `FR-008`, `FR-009`, `FR-021`, `CON-001`, `CON-003`

### 8.2 Model selection

`[설계]` Model selection은 config의 monitor metric과 validation result를 사용한다. No-gradient 모델처럼 하나의
finalized state만 있는 경우 selection checkpoint와 final checkpoint가 같을 수 있다. Test result는 selection에
사용하지 않는다.

Implements: `FR-009`, `CON-009`

### 8.3 Resume

`[설계]` Resume은 model, adapter, optimizer, scheduler, scaler, epoch, global step 및 RNG state를 복원한다.
Auxiliary loader iterator 자체는 저장하지 않고, seed와 global step 또는 model-specific deterministic position으로
재구성한다. Reference equivalence가 정확한 iterator position을 요구하면 해당 position을 adapter state로 보존한다.

Implements: `FR-021`, `FR-022`, `NFR-002`

## 9. Evaluation Flow

### 9.1 Call flow

```text
CLI evaluate
  -> resolve/validate config and offline policy
  -> build eval transform, model, metrics, adapter, dataset, loader
  -> load checkpoint model_state + adapter_state
  -> validate inference_ready and protocol compatibility
  -> model.to(device), adapter.to(device)
  -> Trainer.evaluate under eval/no_grad
       -> adapter.eval_step
       -> raw output conversion
       -> checkpoint-derived post-processing
       -> metric update
  -> aggregate metrics
  -> save metrics and optional visualization
```

`[설계]` Evaluation은 statistics, memory bank 또는 threshold를 test data에서 새로 fit하지 않는다. Reference가
validation과 test를 공유하는 별도 run은 protocol metadata로 명시한다.

Implements: `FR-010`, `FR-012`, `FR-016`, `FR-017`, `FR-021`, `CON-009`

### 9.2 Output

`[설계]` 기존 `metrics_final.json`, visualization 및 log convention을 유지한다. Metric output은 level과 처리
상태를 구분할 수 있어야 한다.

```text
raw/reference-independent: image AUROC, pixel AUROC, optional AUPRO
calibrated: image F1, pixel F1, threshold-dependent fields
protocol: split id, post-processing id, metric implementation id
```

최종 JSON key의 상세 schema는 구현 전에 기존 leaderboard compatibility를 확인해 확정한다.

Implements: `FR-016`, `FR-020`, `FR-022`, `NFR-008`

## 10. Prediction Flow

### 10.1 Input

`[설계]` 현재 CLI convention인 단일 image path 또는 directory를 유지한다. Eval transform은 checkpoint와 resolved
model config에 호환되어야 한다.

### 10.2 Output

`[설계]` 각 sample prediction은 다음 의미를 제공한다.

```text
source path
sample identifier
anomaly score
anomaly map artifact or reference, when supported
predicted image label, when calibrated
predicted mask artifact or reference, when supported and calibrated
image/pixel threshold, when applicable
protocol/checkpoint identity
```

JSON에 대형 anomaly map을 직접 넣는 대신 기존 artifact output convention을 사용한다. Visualization은 동일한
post-processed map과 threshold를 입력으로 사용해야 한다.

Implements: `FR-011`, `FR-012`, `FR-013`, `FR-017`, `FR-022`

### 10.3 Uncalibrated behavior

`[설계]` Reference protocol이 threshold를 사용하지 않거나 checkpoint가 명시적으로 uncalibrated이면 score/map은
반환할 수 있지만 label/mask는 임의 threshold로 생성하지 않는다. 결과에 uncalibrated 상태를 명시한다.

Implements: `FR-011`, `NFR-011`
## 11. Benchmark and Reference Equivalence

### 11.1 Benchmark 유형

`[설계]` 기존 benchmark runner를 유지하되 목적을 결과 metadata에서 구분한다.

- smoke/comparative run: pipeline과 여러 모델의 제한된 조건 비교
- reference-equivalence run: 하나의 integration run을 대응 anomalib reference run과 비교

Reference-equivalence에서는 모델 간 동일 optimizer/transform을 강제하지 않는다. 각 모델은 자신의 reference
protocol과 동등해야 하며, cross-model control의 차이는 승인된 exception으로 기록한다.

Implements: `FR-018`, `FR-019`, `FR-020`, `GAP-010`

### 11.2 Reference manifest

`[설계]` Reference run과 integration run은 다음 필드를 포함한 immutable manifest로 연결한다.

| 영역 | 필수 기록 |
|---|---|
| Source | anomalib repository, version/commit, model source paths, local integration revision |
| Model | model name/family, model size, backbone/layers, algorithm parameters |
| Dataset | name, release/version/checksum, category, root identity, split manifest |
| Assets | weight filename, source, checksum, load target, auxiliary dataset identity |
| Runtime | seed, device, AMP, determinism, Python/PyTorch/torchvision/metric library versions |
| Input | image size, resize/crop, normalization, augmentation, batch size, drop-last |
| Optimization | trainable parameters, optimizer, learning rate, weight decay, scheduler, cadence |
| Budget | epochs, max steps, early stopping, validation cadence |
| Output processing | map interpolation, smoothing, normalization, quantiles, thresholds |
| Metrics | name, implementation/version, fields, aggregation, FPR limit 등 parameters |
| Hardware | GPU, CUDA, driver, relevant deterministic warnings |

Implements: `FR-019`, `FR-020`, `FR-022`, `NFR-001`, `NFR-002`, `CON-008`

### 11.3 Protocol diff

`[설계]` Runner는 reference manifest와 resolved integration config를 비교해 field별 상태를 기록한다.

```text
equal
equivalent with documented translation
different with approved reason
missing / invalid
```

필수 field가 missing이거나 승인되지 않은 차이가 있으면 equivalence 결과를 pass로 판정하지 않는다.

Implements: `FR-020`, `NFR-008`, `NFR-011`

### 11.4 Tolerance 결정

`[미확정]` 수치를 임의로 정하지 않는다. Target revision과 protocol을 고정한 뒤 동일 reference run을 승인된
횟수만큼 반복해 hardware/library 환경에서의 분산을 측정한다. Integration tolerance는 reference 반복 분산,
metric 해상도 및 수치 오차를 근거로 사용자 승인 후 manifest에 기록한다.

Implements: `NFR-001`, `NFR-012`, `AC-008`, `GAP-013`

### 11.5 Failure isolation

`[설계]` 현재 benchmark의 per-split exception isolation을 유지한다. Failed row는 exception type, message,
protocol identity, partial artifact path 및 incomplete state를 기록하며 다른 independent run을 삭제하지 않는다.

Implements: `FR-025`, `NFR-008`, `AC-020`

## 12. Metrics and Post-processing

### 12.1 처리 단계

| 단계 | 책임 | Stateful 여부 |
|---|---|---|
| Raw model output | upstream pure model | model state |
| Output conversion | per-model adapter | stateless |
| Algorithm map normalization | upstream/per-model integration | model state 가능 |
| Protocol post-processing | anomaly task integration | adapter state 가능 |
| Threshold application | anomaly task integration | adapter state |
| Metric aggregation | registered metric | run-local state |

Implements: `FR-013`, `FR-016`, `FR-017`, `FR-021`

### 12.2 Metric 정의

`[설계]` Metric은 reference와 동일한 input field, interpolation, mask inclusion, threshold 및 aggregation을 사용한다.

- image AUROC: continuous `pred_score`와 binary label
- pixel AUROC: continuous anomaly map과 모든 evaluation image의 binary mask
- AUPRO/PRO: reference와 같은 connected-component와 FPR limit
- F1: validation에서 결정된 threshold를 test에 고정 적용
- threshold-free metric에 thresholded prediction을 입력하지 않음

`[미확정]` AUPRO/PRO와 F1의 초기 필수 포함 여부 및 metric implementation source는 target reference 선정 후
확정한다.

Implements: `FR-016`, `NFR-001`, `GAP-008`

### 12.3 Calibration

`[설계]` Threshold, min/max, quantile 및 score distribution state는 calibration source split과 함께 저장한다.
Calibration은 idempotent해야 하며 checkpoint restore 후 test/predict에서 재계산하지 않는다.

Implements: `FR-012`, `FR-017`, `FR-021`, `CON-009`

### 12.4 Score/map shape

`[설계]` Pixel metric 직전 anomaly map과 mask의 shape가 같아야 한다. Shape가 다르면 silent resize하지 않고,
reference에서 승인된 interpolation rule이 있을 때만 명시적으로 변환한다.

Implements: `FR-013`, `FR-014`, `NFR-011`

## 13. Configuration

### 13.1 기존 config 유지

`[설계]` 현재 top-level `meta`, `runtime`, `data`, `model`, `loss`, `metrics`, `adapter`, `optim`, `train`, `output`
구조와 `_base` merge/override를 유지한다. 새 config framework를 만들지 않는다.

Implements: `FR-001`, `CON-003`, `NFR-010`

### 13.2 Generic과 model-specific 경계

| 영역 | Generic config | Model-specific config |
|---|---|---|
| Runtime | device, seed, AMP, determinism, network policy | reference가 요구하는 제한 override |
| Data | root, split, workers, primary batch | normalization, batch constraint, auxiliary data spec |
| Model | registry name | backbone, layers, model size, local weights |
| Optimization | optimizer/scheduler schema | parameter selection과 reference values |
| Train | epochs/max steps, monitor | validation preparation 및 cadence requirement |
| Metric | metric list | reference-required parameters |
| Reference | manifest identity | upstream model/config/source mapping |

User는 model config 선택으로 내부 차이를 받으며 auxiliary iterator나 parameter group을 직접 조립하지 않는다.

Implements: `FR-006`, `FR-007`, `FR-008`, `FR-022`

### 13.3 Validation rules

`[설계]` Config validation은 construction 전에 다음을 검사한다.

- optimizer-required lifecycle에 optimizer가 존재하는가
- optimizer-free lifecycle에 잘못된 scheduler가 없는가
- scheduler cadence와 max steps/epochs가 유효한가
- required auxiliary loader와 local asset이 존재하는가
- model-specific input constraint가 충족되는가
- monitor metric이 선언되었는가
- reference manifest의 필수 field가 있는가
- split protocol이 명시되었는가

Implements: `FR-006`, `FR-008`, `FR-023`, `NFR-011`

### 13.4 Reference metadata

`[설계]` Benchmark config는 reference manifest를 path 또는 immutable identifier로 참조한다. Resolved run output에는
manifest snapshot과 protocol diff를 저장한다.

Implements: `FR-018`, `FR-019`, `FR-020`, `FR-022`

## 14. Checkpoint and State Management

### 14.1 Ownership

`[설계]` Checkpoint file I/O는 기존 core checkpoint module이 소유한다. Model과 adapter는 serializable state만
제공한다.

### 14.2 Checkpoint fields

기존 field를 유지하고 다음을 추가한다.

```text
adapter_state
global_step
lifecycle_state or equivalent readiness evidence
protocol/reference identity
checkpoint schema version
```

Model-specific tensor state는 우선 `model.state_dict`에 포함한다. Adapter state는 tensor와 JSON-compatible scalar를
지원하되 device-independent하게 restore되어야 한다.

Implements: `FR-012`, `FR-021`, `FR-022`, `NFR-002`

### 14.3 Save timing

`[설계]` 두 checkpoint 의미를 구분한다.

- selection state: validation monitor로 선택된 training state
- inference-ready state: 선택 state에 final calibration/finalization을 적용한 state

사용자-facing evaluate/predict는 inference-ready checkpoint를 기본으로 한다. 현재 `best.pth` convention을
유지할지 별도 이름을 사용할지는 backward compatibility 확인 후 결정한다.

Implements: `FR-009`, `FR-010`, `FR-012`, `FR-021`

### 14.4 Restore validation

`[설계]` Load 시 schema, model identity, adapter identity, required state, source revision 및 config compatibility를
검사한다. Adapter state가 필수인 calibrated model에서 누락되면 uncalibrated로 조용히 진행하지 않는다.

Implements: `FR-010`, `FR-012`, `FR-023`, `NFR-011`

### 14.5 State round-trip

`[설계]` 동일 input에 대해 save 전과 새 process restore 후의 raw output, calibrated score/label 및 map/mask를
허용 수치 오차 내에서 비교한다.

Implements: `AC-010`, `NFR-002`

## 15. Error Handling and Validation

### 15.1 Early errors

| 오류 | 검출 시점 | 동작 |
|---|---|---|
| unsupported registry name | config validation | available entries와 함께 실패 |
| incompatible model/adapter | construction | expected contract와 실제 identity를 보고 |
| missing local weight | asset validation | exact path와 asset id를 보고 |
| weight key/checksum mismatch | model construction | strict failure, random fallback 금지 |
| invalid dataset/category structure | dataset construction | expected directory/sample을 보고 |
| missing abnormal mask | dataset validation | sample id/path와 함께 실패 |
| invalid split or leakage | split validation | overlapping ids와 protocol을 보고 |
| missing auxiliary loader | pre-fit validation | required loader name을 보고 |
| optimizer/loss mismatch | pre-fit validation | lifecycle requirement를 보고 |
| non-ready checkpoint | evaluate/predict load | missing state와 preparation requirement를 보고 |
| reference manifest mismatch | benchmark preflight | field-level protocol diff를 보고 |
| metric shape/label mismatch | evaluation preflight | score/target shape와 metric id를 보고 |

Implements: `FR-023`, `FR-025`, `NFR-011`, `CON-007`

### 15.2 Exception boundary

`[설계]` CLI 단일 실행은 user-facing error로 실패한다. Benchmark는 현재처럼 독립 run boundary에서 exception을
failed result로 변환하며 traceback/log를 보존한다.

Implements: `FR-025`, `AC-020`

## 16. Offline / Local Asset Behavior

### 16.1 Network policy

`[설계]` 현재 process-start offline guard와 environment 설정을 유지한다. Third-party model constructor 및
dataset 준비 함수가 호출되기 전에 guard가 활성화되어야 한다.

Implements: `NFR-006`, `CON-006`

### 16.2 Local resolution

`[설계]` 모든 필수 asset은 resolved config 또는 asset manifest의 explicit absolute/local path로 resolution한다.
Environment default가 있더라도 resolved config에 최종 path를 기록한다.

Implements: `FR-022`, `FR-023`, `NFR-006`

### 16.3 Implicit download 방지

`[설계]` Architecture 생성은 `weights=None`, `pretrained=False` 또는 동등한 no-download mode를 사용한다. 이후
local state를 strict load한다. anomalib datamodule의 `prepare_data`, EfficientAD download helper 및 `torch.hub`
download path는 product runtime에서 호출하지 않는다.

Implements: `CON-002`, `CON-006`, `CON-007`, `OOS-011`

### 16.4 Asset manifest

`[설계]` Asset entry는 id, type, local path, source URL 또는 provenance, checksum, expected format, consumer 및
optional/required 상태를 가진다. Directory asset은 version marker 또는 deterministic file manifest를 사용한다.

Implements: `FR-008`, `FR-022`, `FR-023`, `NFR-002`

### 16.5 Optional online behavior

`[설계]` 향후 optional download 기능이 생겨도 offline core requirement와 분리되고 명시적 opt-in이어야 한다.
현재 범위에서는 구현하지 않는다.

Implements: `CON-006`, `OOS-011`

## 17. Testing Strategy

### 17.1 Unit tests

- upstream output type/shape에서 common anomaly output으로의 conversion
- image/mask transform alignment와 binary mask
- normal evaluation image의 zero mask
- optimizer-present/optimizer-none train-step validation
- trainable parameter selection과 scheduler cadence
- validation hook order
- auxiliary loader resolution
- metric input, reset, aggregation
- threshold/normalization state save/load
- missing/mismatched asset error
- reference manifest diff

Implements: `NFR-007`, `AC-006`, `AC-007`, `AC-012`, `AC-019`

### 17.2 Integration tests

- Dataset → collate → model → adapter → train/evaluate
- gradient lifecycle
- no-gradient collection/finalize lifecycle
- auxiliary loader lifecycle
- checkpoint save → new process load → evaluate/predict
- calibrated prediction과 visualization input
- benchmark failure isolation
- existing Classification/Segmentation/Detection regression

Implements: `AC-001`, `AC-004`, `AC-010`, `AC-016`, `AC-018`, `AC-020`

### 17.3 Smoke tests

`[설계]` 빠른 smoke는 local fixture 또는 승인된 small category/subset과 짧은 budget을 사용한다. 목적은 shape,
state transition, offline behavior 및 command completion이며 reference accuracy pass를 주장하지 않는다.

Implements: `FR-001`, `NFR-007`

### 17.4 Reference benchmark tests

`[설계]` Long-running reference test는 CI fast suite와 분리한다.

- pinned anomalib reference run
- same/equivalent integration run
- protocol diff가 승인 상태인지 확인
- 반복 reference 결과로 확정한 tolerance 적용
- raw metric/result/config/environment 보존

Implements: `FR-019`, `FR-020`, `NFR-001`, `NFR-002`, `NFR-012`, `AC-008`, `AC-009`, `AC-013`, `AC-014`

### 17.5 Static/purity tests

- core engine/CLI의 task명·model명 conditional 탐지
- Lightning/anomalib Engine runtime import 탐지
- implicit download API 탐지
- vendored source manifest와 local diff 검증
- license/notice 존재 검증

Implements: `NFR-003`, `NFR-004`, `CON-002`, `CON-005`, `CON-011`, `AC-002`, `AC-003`, `AC-017`

## 18. Migration from Legacy Repositories

| Legacy component | Action | Target | Reason |
|---|---|---|---|
| anomalib-derived `torch_model.py` | reference/compare, target revision에서 재취득 우선 | per-model algorithm area | legacy exact revision이 고정되지 않음 |
| `loss.py`, `anomaly_map.py` | reference/compare, upstream fidelity 검증 | per-model algorithm area | algorithm 의미 보존 |
| local `TimmFeatureExtractor` adaptation | adapt candidate | model component/offline loading | no-download architecture와 local weight 경험 |
| STFPM trainer | reference-only lifecycle evidence | per-model adapter specification | student-only SGD와 loss invocation 지식 |
| EfficientAD trainer | reference-only lifecycle evidence | per-model adapter specification | auxiliary loader, statistics, quantile 지식 |
| PatchCore/PaDiM trainer | reference-only lifecycle evidence | generic no-optimizer capability | collect/finalize 필요성 증거 |
| MVTec/VisA/BTAD Dataset | selective reference | current Dataset registry | parsing 규칙만 재사용, contract는 current 우선 |
| backbone filename mapping | adapt into asset manifest | offline asset validation | environment magic 제거와 provenance 필요 |
| `BaseTrainer` | remove | none | current `Trainer`와 중복 |
| `Evaluator`/Predictor | remove | none | current adapter/engine/CLI와 중복 |
| Factory | remove | none | current registry와 중복 |
| Config merge | remove | none | current config system과 중복 |
| EarlyStopper | reference only if approved protocol requires | current training config/capability | legacy framework 보존 불필요 |
| experiment loop scripts | remove | current benchmark runner | orchestration 중복 |
| `anomaly_detection_dev` Phase framework | reference-only | tests/document evidence | 구현 architecture source가 아님 |

Implements: `FR-003`, `FR-004`, `FR-024`, `NFR-010`, `OOS-004`

## 19. Dependency Changes

### 19.1 현재 확인

`[확인]` 현재 requirements는 PyTorch, torchvision, torchmetrics, PyYAML, Pillow 및 기존 task dependency를
포함하고 `timm`과 anomalib는 포함하지 않는다.

### 19.2 후보 dependency

| Candidate | 필요 근거 | 기존 dependency 대체 가능성 | Scope | 결정 상태 |
|---|---|---|---|---|
| `timm` | 분석한 STFPM upstream feature extractor | torchvision model injection으로 가능한지 target revision 검증 필요 | runtime | 미확정 |
| `safetensors` | 선택 backbone asset format이 요구할 수 있음 | `.pth`만 승인하면 불필요 | runtime optional | 미확정 |
| connected-component/AUPRO 지원 | reference metric이 AUPRO를 요구할 수 있음 | torch/torchmetrics로 동일 정의 구현 가능한지 검증 필요 | runtime/test | 미확정 |
| anomalib package | reference run에는 필요 | product runtime에는 허용하지 않음 | benchmark reference environment only | runtime 금지 |
| Lightning | anomalib reference run의 transitive dependency | product runtime에는 불필요 | reference environment only | runtime 금지 |

새 dependency는 target model/metric을 확정한 뒤 사용자 승인과 offline package availability를 확인한다. Reference
environment dependency와 product runtime dependency를 분리한다.

Implements: `NFR-014`, `CON-002`, `CON-006`, `GAP-012`

## 20. PRD Traceability Matrix

### 20.1 Functional Requirements

| PRD ID | Requirement | SPEC Section | Verification |
|---|---|---|---|
| FR-001 | 공통 실행 workflow | 2.1, 4.1, 8, 9, 10, 11 | command integration tests |
| FR-002 | Anomaly task 통합 | 2.3, 3, 4.1 | registry/construction test |
| FR-003 | anomalib pure model 재사용 | 4.5, 6.1, 6.2, 18 | source manifest/diff test |
| FR-004 | 최소 adaptation | 4.5, 6.1, 6.3, 18 | upstream diff review |
| FR-005 | 이질 lifecycle | 3.3, 7, 8 | gradient/no-gradient integration tests |
| FR-006 | model optimization | 3.3, 6.7, 7.3, 8 | parameter/cadence tests |
| FR-007 | model preprocessing | 5.4, 13.2 | resolved transform tests |
| FR-008 | auxiliary asset/data | 5.5, 7.2, 8, 16.4 | auxiliary loader integration test |
| FR-009 | 학습 결과 선택 | 5.3, 7.4, 8.2, 14.3 | leakage/selection tests |
| FR-010 | 평가 | 9, 14.4 | checkpoint evaluation test |
| FR-011 | prediction | 4.3, 10 | single/directory prediction tests |
| FR-012 | calibrated prediction | 9, 10, 12.3, 14 | checkpoint round-trip test |
| FR-013 | anomaly output semantics | 4.2~4.4, 10.2, 12 | output conversion tests |
| FR-014 | MVTec | 5.2, 5.4, 12.4 | dataset fixture/integration test |
| FR-015 | dataset 독립성 | 3.2, 4.4, 5.1, 5.6 | alternate dataset contract test |
| FR-016 | anomaly metric | 4.4, 9.2, 12.2 | metric equivalence tests |
| FR-017 | post-processing | 4.5, 10, 12 | state/threshold tests |
| FR-018 | benchmark orchestration | 2.7, 11, 13.4 | benchmark integration test |
| FR-019 | reference equivalence | 11, 17.4 | paired benchmark |
| FR-020 | protocol difference | 11.2~11.3, 13.4 | manifest diff test |
| FR-021 | checkpoint 완전성 | 6.8, 8.3, 12.3, 14 | round-trip test |
| FR-022 | 재현 정보 | 5.3, 11.2, 13.4, 14, 16.4 | artifact schema test |
| FR-023 | local asset 검증 | 6.4, 13.3, 15, 16 | offline/missing asset tests |
| FR-024 | 새 모델 추가 | 2.2, 3.2, 4.1, 6.3 | second-lifecycle integration review |
| FR-025 | 실패 격리 | 11.5, 15.2 | injected benchmark failure |

### 20.2 Non-Functional Requirements

| PRD ID | Requirement | SPEC Section | Verification |
|---|---|---|---|
| NFR-001 | reference 성능 | 11, 12.2, 17.4 | tolerance benchmark |
| NFR-002 | 재현성 | 5.3, 8.3, 11.2, 14, 16.4, 17.4 | repeated run/round-trip |
| NFR-003 | upstream fidelity | 4.5, 6.1~6.2, 17.5 | manifest/diff/static review |
| NFR-004 | task-agnostic engine | 2.3, 3.2, 7.1, 17.5 | purity scan/regression |
| NFR-005 | 확장성 | 3.2~3.3, 7.1, 17.2 | additional lifecycle test |
| NFR-006 | offline | 5.5, 6.4, 16 | network-blocked lifecycle |
| NFR-007 | testability | 17 | layered test evidence |
| NFR-008 | 관찰 가능성 | 2.9, 9.2, 11.3, 11.5, 15.2 | artifact/failure inspection |
| NFR-009 | dataset 독립성 | 3.2, 4.4, 5 | contract review |
| NFR-010 | 유지보수성 | 2.2, 3, 6.3, 18 | responsibility review |
| NFR-011 | 명시성 | 7.5, 10.3, 11.3, 13.3, 14.4, 15 | negative tests |
| NFR-012 | 수치 현실성 | 11.4, 17.4 | repeated reference statistics |
| NFR-013 | 기존 task 회귀 | 17.2 | existing task regression |
| NFR-014 | 최소 dependency | 19 | dependency review/offline install check |

### 20.3 Constraints

| PRD ID | Constraint | SPEC Section | Verification |
|---|---|---|---|
| CON-001 | pure PyTorch runtime | 3.3, 8 | dependency/runtime inspection |
| CON-002 | Lightning/Engine 금지 | 6.2, 16.3, 17.5, 19 | import/static scan |
| CON-003 | boilerplate lifecycle 소유 | 2, 3.1, 8, 14 | architecture review |
| CON-004 | algorithm 의미 보존 | 4.5, 6.1, 6.5 | upstream diff/model parity |
| CON-005 | engine model-name 분기 금지 | 3.2~3.3, 7.1, 17.5 | purity scan |
| CON-006 | local asset 우선 | 5.5, 6.4, 16, 19 | offline test |
| CON-007 | silent fallback 금지 | 6.4, 14.4, 15, 16.3 | negative asset test |
| CON-008 | reference revision 고정 | 6.1~6.2, 11.2, 13.4 | manifest validation |
| CON-009 | test leakage 금지 | 5.2~5.3, 7.4, 8.2, 9.1, 12.3 | split access test |
| CON-010 | split protocol 명시 | 5.2~5.3, 13.3 | split manifest test |
| CON-011 | license/attribution | 6.1~6.2, 17.5 | notice/source audit |
| CON-012 | 문서 우선순위 | 1, 21 | document review/user gate |

### 20.4 Out of Scope guards

| PRD ID | Guard | SPEC enforcement |
|---|---|---|
| OOS-001 | anomalib 전체 재구현 금지 | 1.4, 6.2, 19 |
| OOS-002 | 새 algorithm 연구 제외 | 1.4, 6.1 |
| OOS-003 | upstream 대규모 rewrite 제외 | 6.1 |
| OOS-004 | legacy framework 개선 제외 | 3.3, 6.3, 18 |
| OOS-005 | anomalib API compatibility 제외 | 6.2, 14 |
| OOS-006 | 단일 training step 강제 제외 | 3.3, 7 |
| OOS-007 | 모든 모델 즉시 지원 제외 | 7.2, 21 |
| OOS-008 | 모든 dataset 즉시 지원 제외 | 5.6, 21 |
| OOS-009 | Enterprise MLOps 제외 | 2.9 |
| OOS-010 | Bitwise 동일성 제외 | 11.4 |
| OOS-011 | 자동 download service 제외 | 16.3~16.5 |

### 20.5 Acceptance Criteria

| PRD ID | Acceptance | SPEC Section | Verification |
|---|---|---|---|
| AC-001 | 공통 command | 8~11, 17.2 | initial model command matrix |
| AC-002 | upstream 추적 | 6.1~6.2, 17.5 | source audit |
| AC-003 | Lightning 비의존 | 6.2, 17.5, 19 | runtime/import scan |
| AC-004 | 서로 다른 lifecycle | 7, 17.2 | gradient + non-standard model |
| AC-005 | model protocol | 5.4~5.5, 6.7, 11.2, 13 | resolved protocol inspection |
| AC-006 | dataset contract | 5, 17.1 | MVTec dataset tests |
| AC-007 | metric | 4.4, 12, 17.1 | reference metric fixtures |
| AC-008 | 성능 재현 | 11.4, 17.4 | approved tolerance benchmark |
| AC-009 | protocol 진단 | 11.3, 17.4 | forced diff report |
| AC-010 | checkpoint round-trip | 14.5, 17.2 | new-process parity |
| AC-011 | offline lifecycle | 16, 17.2 | blocked-network commands |
| AC-012 | missing asset | 15, 17.1 | negative tests |
| AC-013 | 재현성 기록 | 11.2, 13.4, 16.4 | artifact schema test |
| AC-014 | 반복 재현성 | 11.4, 17.4 | repeated runs |
| AC-015 | leakage 방지 | 5.3, 7.4, 9.1 | split access instrumentation |
| AC-016 | 새 모델 비용 | 3.2~3.3, 6.3, 17.2 | integration change review |
| AC-017 | engine 순수성 | 3.2, 7.1, 17.5 | conditional scan |
| AC-018 | 기존 task 회귀 | 17.2 | regression suite |
| AC-019 | 단계적 검증 | 17 | test evidence matrix |
| AC-020 | failure isolation | 11.5, 15.2 | injected failure benchmark |

### 20.6 Current Gap closure mapping

| Gap ID | Closure section | Evidence required |
|---|---|---|
| GAP-001 | 6.1~6.3, 18 | pinned upstream source and diff |
| GAP-002 | 3.3, 7.3, 8 | optimizer-none integration test |
| GAP-003 | 6.7, 13.2 | STFPM parameter selection test |
| GAP-004 | 5.4, 13.2 | EfficientAD input protocol test |
| GAP-005 | 5.5, 7.2, 8 | auxiliary loader test |
| GAP-006 | 7.2~7.4 | memory/statistics lifecycle test |
| GAP-007 | 10, 12.3, 14 | checkpoint round-trip |
| GAP-008 | 12.2, 17.4 | selected metric parity |
| GAP-009 | 5.2~5.3, 11.2, 21 | approved split manifest |
| GAP-010 | 11.1~11.3 | equivalence benchmark artifact |
| GAP-011 | 11.2, 13.4, 16.4 | provenance completeness test |
| GAP-012 | 6.2, 6.4, 19, 21 | dependency/asset decision |
| GAP-013 | 11.4, 17.4, 21 | repeated reference baseline |

## 21. Open Questions / Deferred Decisions

| ID | Question | Why it matters | Options | Recommended direction | Needed before PLAN defines |
|---|---|---|---|---|---|
| OQ-001 | 실제 current source는 어디인가 | 현재 workspace에는 source가 없어 uncommitted/local 차이를 검증하지 못함 | remote `71261cef` 사용 / local repo 제공 | 실제 구현 대상 checkout을 제공하고 revision 고정 | 모든 구현 범위 |
| OQ-002 | target anomalib revision은 무엇인가 | source, dependency, lifecycle, metric이 version별로 다름 | release tag / commit | 검증 가능한 commit hash 고정 | model/source 및 benchmark 작업 |
| OQ-003 | 최초 모델 집합은 무엇인가 | AC-004가 서로 다른 lifecycle을 요구 | STFPM+EfficientAD+PatchCore/PaDiM 중 선택 | gradient/auxiliary와 no-gradient lifecycle을 모두 포함 | model integration 범위 |
| OQ-004 | 최초 MVTec category 범위는 무엇인가 | 실행 비용과 metric tolerance에 영향 | bottle / 일부 category / 15 category | bottle로 pipeline 검증 후 reference 범위 확대가 합리적이나 사용자 확정 필요 | dataset/benchmark 범위 |
| OQ-005 | MVTec validation/test protocol은 무엇인가 | current disjoint 33/50과 anomalib default same-as-test가 다름 | strict disjoint / exact reference / 두 run 모두 | correctness reference와 leakage-safe final evaluation을 별도 protocol로 기록하는 방향 권장 | split manifest와 acceptance benchmark |
| OQ-006 | metric 초기 필수 세트는 무엇인가 | implementation dependency와 acceptance가 달라짐 | AUROC only / AUPRO/F1 포함 | target reference가 보고하는 metric을 필수로 선택 | metric 및 tolerance 작업 |
| OQ-007 | tolerance와 반복 횟수는 무엇인가 | AC-008 pass/fail 기준 | 고정 수치 / repeated baseline 기반 | target environment 반복 결과로 승인 | long benchmark execution |
| OQ-008 | upstream source transport 방식은 무엇인가 | update/diff/license 관리에 영향 | vendor / sync script / narrow package dependency | product runtime에서 anomalib 전체 dependency 없이 immutable source+diff가 남는 방식 | source integration 작업 |
| OQ-009 | `timm`을 허용하는가 | STFPM upstream fidelity와 offline package 가용성에 영향 | runtime dependency / module injection adaptation | target source diff가 더 작은 선택을 dependency 비용과 함께 승인 | STFPM integration |
| OQ-010 | AUPRO dependency를 어떻게 제공하는가 | exact metric parity에 영향 | upstream narrow port / existing libraries / 별도 verified implementation | reference fixture parity를 먼저 비교 | metric integration |
| OQ-011 | local asset inventory와 checksum은 무엇인가 | offline preflight와 reproducibility에 필요 | 현재 파일 조사 / 외부 준비 | 실제 사용 environment에서 manifest 작성 | model construction/benchmark |
| OQ-012 | checkpoint naming/backward compatibility 정책은 무엇인가 | selection state와 inference-ready state 구분 필요 | `best.pth` 갱신 / 별도 finalized checkpoint | 기존 CLI 기본을 깨지 않되 의미를 명시 | checkpoint implementation |
| OQ-013 | reference run environment를 분리할 수 있는가 | product runtime에는 anomalib/Lightning을 넣을 수 없음 | 별도 environment / 외부 결과 import | pinned 별도 reference environment 권장 | reference baseline |
| OQ-014 | full benchmark compute budget은 얼마인가 | STFPM/EfficientAD reference budget이 smoke보다 큼 | full official / capped equivalent / 외부 baseline | smoke와 acceptance benchmark를 분리 | PLAN의 verification 범위 |

Implements: `CON-008`, `CON-012`, `NFR-001`, `NFR-012`, `NFR-014`

## 22. Implementation Impact Summary

### 22.1 Existing files likely modified

`[설계 영향]` 실제 checkout 확인 후 경로를 다시 검증해야 하지만, revision `71261cef` 기준 예상 영향은 다음과
같다.

- `src/core/adapter.py`: optional optimization/state/validation hook contract
- `src/core/engine.py`: optimizer-none, scheduler cadence, validation hook, adapter state invocation
- `src/core/builders.py`: adapter-aware optimization과 named auxiliary loader construction
- `src/core/config.py`: optional optimizer, auxiliary data, reference metadata validation
- `src/core/checkpoint.py`: adapter/global-step/protocol state
- `src/cli/commands.py`: unified construction과 adapter checkpoint restore
- `src/bench/runner.py`, `control.py`, `leaderboard.py`: equivalence manifest/diff/result fields
- `src/tasks/anomaly/adapter.py`: common output/metric/post-processing behavior
- `src/tasks/anomaly/dataset.py`, `transform.py`, `postprocess.py`: contract 보완
- anomaly configs와 asset manifest

### 22.2 New modules likely required

- pinned upstream pure-model source area 또는 sync metadata
- per-model anomaly adapter/integration modules
- stateful anomaly post-processing state representation
- reference manifest/protocol diff validation
- selected reference metric implementation
- source/license/asset provenance manifest

정확한 module 이름과 directory는 실제 checkout과 OQ 결정 후 확정한다.

### 22.3 Reusable existing modules

- Registry와 `TaskAdapter`
- CLI command surface
- `Trainer.evaluate`/`predict`의 task-agnostic loop
- config inheritance/override
- `RunContext`, seed, device, deterministic mode
- offline guard와 strict local weight loader
- checkpoint container와 RNG capture
- MVTec parsing, split disjoint check, anomaly collate/visualization
- benchmark failure isolation, control, profiling, leaderboard
- JSON/YAML/CSV output utilities

### 22.4 Obsolete legacy modules

Legacy `BaseTrainer`, `Evaluator`, Predictor, Factory, Config, experiment loops 및 output manager는 이식하지 않는다.
Model trainer는 lifecycle evidence로만 사용한다.

### 22.5 High-risk integration points

- upstream model fidelity와 import/dependency closure
- EfficientAD auxiliary loader, normalization, quantile 및 scheduler cadence
- no-gradient model의 finalize 시점
- threshold/calibration checkpoint round-trip
- MVTec reference split과 leakage-safe split의 구분
- metric implementation과 map interpolation parity
- reference manifest completeness와 tolerance evidence
- core 변경 후 기존 task regression
- implicit network access 차단

### 22.6 PLAN readiness gate

PLAN 작성 전에 최소한 OQ-001, OQ-002, OQ-003, OQ-005, OQ-006, OQ-011, OQ-013, OQ-014에 대한 사용자
결정 또는 검증 가능한 사실이 필요하다. 이 결정 전에도 공통 contract 검토는 가능하지만, model integration과
reference benchmark의 완료 조건을 확정할 수는 없다.
