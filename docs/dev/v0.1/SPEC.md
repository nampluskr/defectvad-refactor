# SPEC — Anomaly Detection Integration on `cv_boilerplate`

상위 문서: [BRIEF.md](BRIEF.md) · [PRD.md](PRD.md) · 하위 문서: [PLAN.md](PLAN.md)

분석 기준: `cv_boilerplate@65d5412` (로컬: `/mnt/d/projects/nampluskr/00_review/260818_cv-boilerplate`)

이 문서는 실제 코드를 분석해 v0.1의 기술 결정만 기록한다. 상세 data flow와 error handling은 필요할 때 결정한다.

## 1. 현재 구조 (확인된 사실)

```text
src/
├── core/          adapter.py  engine.py  builders.py  checkpoint.py
│                  config.py   registry.py  offline.py  paths.py  context.py
│                  errors.py   logger.py
├── tasks/anomaly/ adapter.py  dataset.py  metric.py  postprocess.py
│                  transform.py  collate.py  visualize.py
│                  models/ stfpm.py  efficientad.py  custom_ae.py
├── data/          split.py
├── cli/           commands.py  parser.py
├── bench/         runner.py  profile.py  control.py  leaderboard.py
└── utils/         io.py  timing.py
```

이 트리는 cv_boilerplate@65d5412의 것이며, 이 저장소로의 반입은 P0-T06에서 수행한다. 이하 §3·§5·§6이 가리키는 경로는 모두 반입 후 기준이다.

`tasks/` 하위에는 `anomaly/` 외에 `classification/`, `detection/`, `segmentation/`, `toy/`가 같은 구조로 존재한다. v0.1은 `anomaly/`만 다루지만, 공통 계층 변경 시 나머지 4개 task가 함께 영향을 받는다(NFR-003).

핵심 계약:

| 위치 | 계약 |
|---|---|
| `core/adapter.py#TaskAdapter` | `train_step`/`eval_step`/`predict_step` 추상 메서드 + `on_fit_start`/`on_epoch_end`/`extra_final_metrics` 등 선택적 hook |
| `tasks/anomaly/adapter.py#AnomalyAdapter` | `train_step` → `model.train_step(images, targets)` 위임, `eval_step` → `model(images)` 호출 |
| 모델 반환 계약 | `train_step` → `{"loss": Tensor, "loss_dict": dict}` / `forward` → `{"pred_score": (B,), "anomaly_map": (B,H,W)}` |
| `core/offline.py` | `load_local_weights()` 로컬 가중치 로딩, 소켓 수준 네트워크 차단 guard |
| `core/registry.py#MODELS` | 데코레이터 기반 모델 등록 |

## 2. 가장 중요한 결정 — 기존 모델 구현 전면 교체

`src/tasks/anomaly/models/stfpm.py`(145줄)와 `efficientad.py`(240줄)가 이미 존재하지만 **anomalib 원본이 아니다**. 두 파일의 헤더 주석이 이를 명시한다.

> "an original implementation written directly from the method description ... **not a verbatim port** of any specific external repository ... is not reachable from this offline sandbox"

`efficientad.py`는 "autoencoder branch uses a **simplified** encoder-decoder ... rather than the paper's specific upsampling schedule"이라고 적혀 있고, 그 근거로 "**v0.1's acceptance criteria do not require literal architectural fidelity**"를 든다. 이는 PRD의 AC-004·AC-006(reference 성능 재현)과 정면으로 충돌한다. 즉 기존 구현은 성능 재현을 요구사항으로 삼지 않는 전제에서 작성되었으므로, 이번 v0.1의 판정 기준을 충족할 수 없다.

이는 BRIEF 원칙 1 위배 상태다. **두 파일을 anomalib 원본 복사본으로 전면 교체한다.**

| 항목 | 처리 |
|---|---|
| 기존 `stfpm.py`, `efficientad.py` | **삭제하며, 사용하거나 참조하지 않는다** |
| `custom_ae.py` | v0.1 범위 밖. 유지하되 건드리지 않는다 |
| 교체 후 | anomalib 원본은 수정 금지(CON-001) |

기존 구현은 논문 설명만 보고 작성된 것이라 anomalib과의 동등성이 보장되지 않는다. 이를 참고하면 그 오차가 새 구현으로 옮겨간다. **모든 모델 코드는 anomalib에서 복사해 온다.**

## 3. 파일 배치

anomalib 디렉터리 구조를 따르지 않고 이 프로젝트 구조에 맞춘다(BRIEF 원칙 1 예외).

```text
src/tasks/anomaly/
├── upstream/                   # anomalib 원본 — 수정 금지 구역
│   ├── components/             # 모델 공통 의존 components
│   │   ├── feature_extractors/
│   │   │   ├── timm.py
│   │   │   └── utils.py
│   │   └── data/
│   │       ├── generic.py      # dataclasses/generic.py 원본
│   │       └── torch_base.py   # dataclasses/torch/base.py 원본 (InferenceBatch 포함)
│   ├── stfpm/
│   │   ├── torch_model.py
│   │   ├── loss.py
│   │   └── anomaly_map.py
│   └── efficient_ad/
│       └── torch_model.py
├── adapters/                   # 이 프로젝트가 소유
│   ├── stfpm.py
│   └── efficientad.py
└── models/                     # 존치 — custom_ae.py 만 남는다
    └── custom_ae.py
```

`upstream/` 하위는 import 경로 외 수정 금지. 기존 `models/stfpm.py`, `models/efficientad.py`는 삭제하며 **참조하지 않는다**(§2).

#### 복사 시 주의 — 두 가지 함정 (P0-T03에서 실제 import 추적으로 확정)

anomalib v2.3.0 원본을 확인한 결과 두 가지가 드러났다.

1. **`components/__init__.py`를 그대로 복사하면 Lightning이 딸려 온다.** 이 파일 39행이 `from .base import AnomalibModule, ...`를 하고, `components/base/anomalib_module.py`는 `import lightning.pytorch as pl`을 한다(`export_mixin.py`, `memory_bank_module.py`도 같다). 따라서 `upstream/components/`에는 **패키지 `__init__`이 아니라 실제로 필요한 모듈만** 옮기고, import는 모듈 단위로 직접 건다. STFPM에 필요한 것은 `feature_extractors/timm.py`와 그것이 쓰는 `feature_extractors/utils.py`뿐이며, 두 파일 모두 lightning을 import하지 않는다(CON-002).
2. **두 `torch_model.py` 모두 `from anomalib.data import InferenceBatch`를 한다.** `InferenceBatch`는 `data/dataclasses/torch/base.py:26`의 `NamedTuple`이지만, `anomalib.data` 패키지 `__init__`은 `datamodules.base`·`datamodules.depth`·`datamodules.image`·`datamodules.video`를 전부 끌어오고 이들은 `AnomalibDataModule`(lightning `DataModule`)에 닿는다. `anomalib` 패키지 경로를 통해 import하는 한 어떤 하위 모듈을 지정하든 부모 패키지 `__init__`이 먼저 실행되어 Lightning이 함께 들어온다.
   결정: `data/dataclasses/torch/base.py`와 그 유일한 내부 의존인 `data/dataclasses/generic.py`를 **파일째 `upstream/components/data/`로 복사**하고, `torch_model.py` 두 곳의 import만 로컬 경로(`from tasks.anomaly.upstream.components.data.torch_base import InferenceBatch` 형태, 정확한 모듈 경로는 P1-T01에서 확정)로 바꾼다. 두 파일 모두 torch·numpy·torchvision 외 의존이 없고(lightning 없음), `InferenceBatch` 외에 함께 딸려 오는 `ToNumpyMixin`·`DatasetItem`·`Batch`(base.py)는 미사용이지만 파일 단위 복사 원칙(CON-001 — import 경로 외 수정 금지)상 클래스만 추출하지 않고 파일 전체를 그대로 둔다.

`models/` 디렉터리 자체는 삭제하지 않는다. `custom_ae.py`는 v0.1 범위 밖이지만 registry에 `custom_ae_anomaly`로 등록되어 있고 `configs/anomaly/custom_ae.yaml`이 이를 참조하므로, 디렉터리째 삭제하면 기존 config가 깨진다. `models/`에는 `custom_ae.py`와 `__init__.py`만 남는다.

## 4. 통합 계층 — adapter

### 4.1 defectvad wrapper 방식을 쓰지 않는 이유

`AnomalyAdapter.train_step`은 현재 `model.train_step(images, targets)`으로 위임하고, 모델이 loss를 소유한다. 이는 defectvad에서 사용자가 정의했던 wrapper 구조이며, 두 가지 문제가 있다.

1. anomalib 원본 `torch_model.py`에는 `train_step`이 없다. 이 계약을 유지하려면 원본을 감싸는 계층에 모델별 코드가 쌓인다.
2. boilerplate의 상위 개념은 `TaskAdapter`다. `core/engine.py`는 모델을 직접 호출하지 않고 `adapter.train_step` / `adapter.eval_step` / `adapter.predict_step`만 호출한다.

따라서 **모델별 차이는 wrapper가 아니라 adapter가 흡수한다.** 모델 자리에는 anomalib `torch_model.py`의 nn.Module이 그대로 들어간다.

### 4.2 책임 배치

| 책임 | 위치 | 비고 |
|---|---|---|
| 네트워크·알고리즘 | `upstream/` anomalib 원본 | 수정 금지 |
| forward 호출과 출력 변환 | 공통 `AnomalyAdapter` + `postprocess.py#to_output_dict` | anomalib `InferenceBatch` → `{"pred_score", "anomaly_map"}` (아래 참조) |
| loss 계산 | 모델별 adapter | `upstream`의 loss 모듈 호출 |
| lifecycle hook | 모델별 adapter | `on_fit_start` / `on_epoch_end` 등 |
| metric·post-processing | 공통 `AnomalyAdapter` | 기존 코드 재사용 |
| optimizer·scheduler | `configs/` + `core/builders.py` | 코드가 아니라 config로 지정 |
| 모델 생성 | `core/registry.py#MODELS` | 원본 nn.Module을 등록 |

`build_optimizer(config_optim, model)`이 registry 빌더(`sgd`, `step` 등)로 optimizer/scheduler를 만든다. `lightning_model.py`의 `configure_optimizers` 내용은 **코드가 아니라 config 값으로 이관**한다.

#### parameter group의 한계

`build_optimizer`는 단일 parameter group만 만든다.

```python
trainable = (p for p in model.parameters() if p.requires_grad)
return BUILDERS.build(spec["name"], trainable, **spec.get("params", {}))
```

BRIEF는 EfficientAD optimizer를 "student + autoencoder 파라미터"로 명시하는데, teacher를 `requires_grad_(False)`로 freeze하면 위 필터가 그대로 그 집합을 만든다. STFPM도 teacher freeze로 동일하다. 따라서 **v0.1 두 모델은 config만으로 충족되며 `build_optimizer`를 바꿀 필요가 없다.**

다만 서로 다른 lr을 갖는 복수 parameter group을 요구하는 모델은 이 방식으로 표현할 수 없다. v0.1 범위 밖이므로 지금은 다루지 않고, 그런 모델을 만나면 `core/builders.py`에 group 지정 방식을 추가한다(§7).

#### `InferenceBatch` → 출력 계약 변환 (P2에서 확정)

anomalib `torch_model.py`는 eval 모드에서 dict가 아니라 `InferenceBatch`(`NamedTuple`, `upstream/components/data/torch_base.py`)를 반환하고, `pred_score`/`anomaly_map`에 각각 `(B, 1)`/`(B, 1, H, W)`처럼 크기 1인 클래스·채널 축이 남아 있다(§1 계약 표의 `(B,)`/`(B,H,W)`와 다르다). 이 변환은 STFPM 하나만의 문제가 아니라 `InferenceBatch`를 쓰는 모든 upstream 모델에 공통이므로(EfficientAD도 같은 패턴, §4.5), 모델별 adapter마다 반복하지 않고 `tasks/anomaly/postprocess.py#to_output_dict()`에 한 번 둔다.

- `_asdict()`가 있으면(= NamedTuple) dict로 변환한다.
- `pred_score`가 2차원이면 마지막 축을 squeeze, `anomaly_map`이 4차원이면 채널 축을 squeeze한다.
- `custom_ae`처럼 이미 `(B,)`/`(B,H,W)` dict를 반환하는 모델은 조건에 걸리지 않아 그대로 통과한다.

`AnomalyAdapter.eval_step`/`predict_step`과 `postprocess.py#compute_thresholds`가 모두 `model(images)`를 직접 이 함수에 통과시킨 뒤 사용한다. 모델 이름으로 분기하지 않고 반환 타입·shape로만 판단하므로 NFR-005에 저촉되지 않는다.

### 4.3 모델별 adapter 구조

공통 `AnomalyAdapter`(metric·threshold·visualize)를 상속하고, 모델별 차이만 override한다.

```text
AnomalyAdapter                 # 기존: metric, threshold, smooth, visualize
├── StfpmAdapter               # train_step에서 upstream loss 호출
└── EfficientAdAdapter         # + on_fit_start: teacher 통계, auxiliary loader
                               # + on_validation_start: 분위수 calibration (매 epoch)
                               # + on_fit_end: 미보정 시에만 calibration, 그 후 threshold
```

#### `on_validation_start` — 매 epoch calibration

`TaskAdapter`는 task-agnostic hook `on_validation_start(model, loaders, device)`를 제공한다. 기본은 no-op이며, `core/engine.py#Trainer.fit`이 루프 안에서 **각 epoch의 validation 직전**에 호출한다. 나머지 4개 task와 `StfpmAdapter`는 기본 no-op을 그대로 상속한다(NFR-005).

`EfficientAdAdapter`는 이 hook에서 분위수를 재계산한다. anomalib `lightning_model.py`가 `on_validation_start`에서 하는 것과 같은 시점이다.

이 hook 없이 `on_fit_end`에서만 calibration하면, `torch_model.py#compute_maps`가 `is_set(self.quantiles)`일 때만 정규화하므로 학습 중 모든 epoch의 validation이 `amax(0.5 * raw_map_st + 0.5 * raw_map_stae)`로 채점된다. 스케일이 다른 두 map을 정규화 없이 합산한 값이며, 보정 후 score의 단조 변환이 아니다. 결과적으로 `core/engine.py`가 그 metric으로 고르는 `best.pth`가 최종 평가와 다른 score 정의로 선택된다. pixel AUROC는 map의 공간적 순위가 유지되어 이 오류를 가리지만, image score는 map 전체의 단일 `amax`이므로 그대로 드러난다.

#### `on_fit_end` 순서 제약

기존 `AnomalyAdapter.on_fit_end`는 두 가지를 순서대로 수행한다.

1. 모델의 `on_fit_end` 호출 (모델별 calibration)
2. `compute_thresholds(model, loaders["valid"], device, smooth_sigma)` — valid 전용 threshold 결정

`EfficientAdAdapter`가 `on_fit_end`를 override할 때 **반드시 `super().on_fit_end()`를 호출**하고, 모델이 보정된 상태인 것을 threshold 계산보다 **먼저** 보장해야 한다. 분위수는 `forward`가 두 anomaly map을 합성하는 스케일을 바꾸므로, calibration 전에 계산한 threshold는 calibration 후의 score와 비교 대상이 아니다.

`core/engine.py`는 `on_fit_end` 직전에 best checkpoint 가중치를 다시 로드한다. 분위수는 `nn.Parameter`이므로 `model_state`에 함께 저장·복원된다 — 즉 재로드 시점에 **best epoch의 가중치와 그 가중치로 계산된 분위수가 짝을 이뤄 복원된다**. 따라서 `on_fit_end`는 분위수를 무조건 재계산하지 않고 `is_set(model.quantiles)`가 거짓일 때만 계산한다. 무조건 재계산하면 `reduce_tensor_elems`가 2^24 원소를 넘는 map 집합에서 `torch.randperm`으로 새로 표본을 뽑아, 저장되는 모델이 그것을 선택한 metric과 다른 score 정의를 갖게 된다.

calibration은 `torch.random.fork_rng`로 감싼다. `reduce_tensor_elems`가 전역 RNG를 소비하므로, 감싸지 않으면 valid split의 정상 이미지 수가 다음 epoch의 학습 randomness를 바꾼다(AC-009).

### 4.4 STFPM

| 항목 | 값 | 위치 |
|---|---|---|
| optimizer | SGD, lr 0.4, momentum 0.9, weight_decay 0.001 | `configs/` |
| scheduler | 없음 | `configs/` |
| backbone | `${paths.backbone_root}/resnet18-f37072fd.pth` (§6) | `configs/`(경로) + `adapters/stfpm.py` 팩토리(주입) — §4.6 |
| loss | upstream `loss.py` 호출 | `StfpmAdapter.train_step` |
| teacher freeze | 항상 eval() 유지 | upstream 생성자가 설정, adapter가 유지 보장 |

### 4.5 EfficientAD

| 항목 | 값 | 위치 |
|---|---|---|
| optimizer | Adam, lr 1e-4, weight_decay 1e-5 | `configs/` (`adam` 빌더 추가 필요) |
| scheduler | StepLR, 95% 시점 0.1배 | `configs/` (기존 `step` 빌더 사용) — 아래 참조 |
| pretrained teacher | `${paths.backbone_root}/efficientad_pretrained_weights/` (§6) | `configs/`(경로) + `EfficientAdAdapter.on_fit_start`(로드) — §4.6 |
| auxiliary 데이터 | `${paths.dataset_root}/imagenette2` (§6) | `configs/`(경로) + `EfficientAdAdapter`(loader 생성·소비) — 아래 참조 |
| batch size 1, normalization 미사용 | 제약 | `configs/` |

#### StepLR `step_size` 산출

`build_step_scheduler(target, step_size, gamma=0.1)`는 **절대 epoch 수**를 받는다. "전체 학습의 95% 시점"은 `train.epochs`에서 파생되는 값이므로, config에 상수로 적으면 `epochs`를 바꿀 때 조용히 어긋난다.

v0.1은 `step_size`를 config에 직접 기입하고, 해당 config의 `train.epochs`와 짝이 맞는지 주석으로 남긴다. 파생 값을 config에서 표현하는 일반적 방법(`epochs` 참조 문법 등)은 도입하지 않는다 — 한 모델만 요구하는 기능이므로 미룬다 — 두 모델 이상에서 같은 필요가 확인되기 전에는 새 추상화를 추가하지 않는다.

#### auxiliary 데이터 조달 경로

`core/engine.py#Trainer.fit`은 `train_loader`/`valid_loader`만 받고, `adapter.on_fit_start(model, loaders, device)`의 `loaders`도 `{"train", "valid"}`뿐이다. ImageNette auxiliary 스트림은 이 경로로 들어오지 않는다.

**`EfficientAdAdapter`가 스스로 DataLoader를 만든다.** `on_fit_start`에서 config의 auxiliary 경로를 읽어 `core/builders.py#build_dataloader`로 loader를 생성하고, adapter 인스턴스에 보관한 뒤 `train_step`에서 배치를 하나씩 꺼내 쓴다. train_loader보다 짧으면 순환한다.

`core/engine.py`의 시그니처는 바꾸지 않는다. auxiliary 스트림은 EfficientAD 하나만 요구하는 것이므로, 공통 engine에 세 번째 loader 개념을 추가하면 나머지 4개 task가 쓰지 않는 인자를 떠안게 된다(NFR-003·NFR-005).

- adapter가 auxiliary 경로를 읽으려면 adapter params로 전달되어야 한다 → `configs/anomaly/efficientad.yaml`의 `adapter.params`에 둔다.
- auxiliary transform(normalization 미사용 제약 포함)이 `tasks/anomaly/transform.py`에 추가로 필요한지는 upstream `torch_model.py` 확인 후 P4에서 판단한다(§7).

Lightning hook 매핑:

| anomalib `lightning_model.py` | 이 프로젝트 |
|---|---|
| `on_train_start` — teacher 채널 통계, auxiliary 준비 | `EfficientAdAdapter.on_fit_start` |
| `prepare_pretrained_model` — teacher 가중치 다운로드·로드 | `EfficientAdAdapter.on_fit_start` (로컬 경로 strict load, 다운로드 없음 — §4.6) |
| `on_validation_start` — 90%/99.5% 분위수 | `on_fit_end` (또는 `on_epoch_end`) |
| `configure_optimizers` | `configs/` + `core/builders.py` |

Adam 빌더는 현재 registry에 없다(`adamw`, `sgd`만 존재). `core/builders.py`에 `adam`을 추가한다 — 모델명 분기가 아닌 범용 빌더이므로 NFR-005에 저촉되지 않는다.

### 4.6 pretrained 가중치 주입 — no-download 모델 생성

두 모델의 상황이 다르다.

| 모델 | upstream `torch_model.py`의 가중치 로딩 | 처리 |
|---|---|---|
| EfficientAD | 없음. teacher 다운로드·로드는 `lightning_model.py`의 `prepare_pretrained_model()`에 있다 | `EfficientAdAdapter.on_fit_start`에서 로컬 로드. 원칙 2의 "hook은 adapter로 옮겨 적는다"에 그대로 해당한다 |
| STFPM | `STFPMModel.__init__`이 teacher를 `TimmFeatureExtractor(..., pre_trained=True, ...)`로 생성한다 | 생성 시점에 다운로드를 시도하므로 아래 절차가 필요하다 |

STFPM에 대해 세 가지 선택지가 있었고 **A안을 채택한다**(2026-08-20 사용자 결정).

| 안 | 방식 | 판정 |
|---|---|---|
| A | adapter의 팩토리가 no-download 상태로 upstream 생성자를 호출한 뒤 로컬 state_dict를 주입 | **채택**. upstream 무수정, 자산 경로가 config에 남는다 |
| B | timm/HF 캐시를 규격대로 사전 배치하고 `HF_HUB_OFFLINE=1` | 미채택. 코드 변경은 없으나 자산 배치가 config 밖으로 새어 나가고, 캐시 레이아웃 오류가 조용한 다운로드 시도로 나타난다 |
| C | upstream `torch_model.py`의 `pre_trained`를 `False`로 변경 | 미채택. CON-001 위반 |

#### A안 절차

`adapters/stfpm.py`에 팩토리 함수를 두고 `MODELS`에 등록한다. 팩토리가 수행하는 일은 다음과 같다.

1. timm의 pretrained 인자를 `False`로 덮어쓰는 wrapper로 생성 함수를 임시 치환한다. 치환 범위는 upstream 생성자 호출 한 문장으로 한정하고 `try/finally`로 원복한다.
2. 그 안에서 upstream `STFPMModel(layers=..., backbone=...)`을 생성한다. 이 시점에는 teacher·student 모두 랜덤 초기화 상태다.
3. `core/offline.py#load_local_weights`로 로컬 `.pth`를 **teacher feature extractor에만** 적재한다(`model.teacher_model.feature_extractor`). 적재 조건은 아래 `strict load 조건`을 따른다. student는 `pre_trained=False`로 생성되므로 랜덤 초기화를 유지한다 — teacher를 랜덤 시작점에서 따라잡는 것이 방법의 전제다.
4. 파일 부재·키 불일치는 `LocalAssetError`로 즉시 실패시킨다. 조용한 랜덤 초기화 폴백을 두지 않는다(CON-004).
5. teacher freeze(§4.4)는 upstream 생성자가 이미 수행한다. 팩토리는 이를 다시 설정하지 않고, outer `train()` 이후에도 eval이 유지되는지만 adapter에서 보장한다.

이 치환은 upstream 파일을 수정하지 않지만 upstream의 생성 경로에 개입한다. 따라서 **`adapters/` 밖으로 내보내지 않는다.** `core/`에 일반화된 "pretrained 차단 컨텍스트"를 만들지 않는다 — 아직 한 모델만 요구하는 기능이고, 공통 계층에 모델 전용 개념을 남기게 된다(NFR-005).

치환 지점은 아래 레거시 확인 결과로 `timm.create_model`로 확정했다. upstream 복사(P1-T01) 후 anomalib 원본이 defectvad가 참조한 버전과 같은 형태인지만 재확인한다.

#### 레거시 defectvad 확인 결과

`defectvad@14879ea`(로컬: `/mnt/d/archive/_inbox/github/defectvad`)는 **`torch_model.py`를 수정하지 않았다.** `models/stfpm/torch_model.py:96`은 `TimmFeatureExtractor(backbone=self.backbone, pre_trained=True, layers=layers).eval()`로 원본 그대로이고, 바뀐 것은 37~38행의 import 경로뿐이다.

대신 **`components/feature_extractor.py`의 `TimmFeatureExtractor`를 고쳤다.** `timm.create_model(..., pretrained=False, pretrained_cfg=None, features_only=True, ...)`로 생성한 뒤, `pre_trained` 인자가 참이면 `common/backbone.py#get_backbone_path`가 돌려준 로컬 경로를 `load_state_dict(state_dict, strict=False)`로 적재한다. 경로는 `BACKBONE_DIR` 환경변수와 backbone 이름→파일명 매핑 표(`BACKBONE_WEIGHT_FILES`)로 결정한다.

EfficientAD는 예상대로 `torch_model.py` 밖이다. `models/efficientad/model_trainer.py#prepare_pretrained_model`이 anomalib `lightning_model.py`의 같은 이름 메서드를 옮겨 적은 것으로, 다운로드 호출은 주석 처리하고 `self.model.teacher.load_state_dict(torch.load(teacher_path, ...))`만 남겼다.

이 프로젝트에 그대로 가져올 수는 없다. `upstream/components/`도 CON-001의 수정 금지 구역이므로, defectvad가 택한 위치(components 파일 직접 수정)는 이 프로젝트에서는 C안에 해당한다. 다만 세 가지가 확인되어 A안의 미결정 사항이 해소된다.

| 확인 사항 | 내용 | A안 반영 |
|---|---|---|
| 개입 지점 | `TimmFeatureExtractor.__init__` 안의 `timm.create_model` 호출 | 치환 대상은 `timm.create_model`로 확정한다 |
| 가중치 파일 | timm `resnet18`에 torchvision `resnet18-f37072fd.pth`를 `strict=False`로 적재해 실사용했다 | 키 호환은 확인되었으나 strict 조건은 아래대로 조정한다 |

#### anomalib v2.3.0 원본 대조

`anomalib@091ca6a`(v2.3.0, 로컬: `/mnt/d/projects/clones/anomalib`)에서 확인한 사실이다.

- `models/image/stfpm/torch_model.py:94` — `TimmFeatureExtractor(backbone=self.backbone, pre_trained=True, layers=layers).eval()`. student는 `pre_trained=False`, teacher freeze는 생성자가 수행한다. §4.4 표와 일치한다.
- `models/components/feature_extractors/timm.py:120` — `timm.create_model(backbone, pretrained=pre_trained, pretrained_cfg=None, features_only=True, exportable=True, out_indices=self.idx)`.
- **`pretrained_cfg=None`은 upstream 원본이 이미 넘긴다.** defectvad가 추가한 것이 아니다. 따라서 치환 wrapper가 강제할 값은 `pretrained=False` 하나다.
- `_map_layer_to_idx`(같은 파일 147행)도 `timm.create_model`을 호출하지만 `pretrained=False`이므로 다운로드 경로가 아니다. 치환 wrapper는 이 호출도 함께 지나가되 동작을 바꾸지 않는다.
- `models/image/efficient_ad/lightning_model.py:162` — `prepare_pretrained_model()`이 `Path("./pre_trained/")` 아래를 보고 없으면 `download_and_extract`를 호출한다. 이 메서드는 `on_train_start`(377행)에서 불린다. `EfficientAdAdapter.on_fit_start`가 **다운로드 분기 없이** `self.model.teacher.load_state_dict(...)`에 해당하는 부분만 옮겨 적는다.

`BACKBONE_DIR` 환경변수를 유일한 조달 수단으로 쓰는 방식은 채택하지 않는다. 이 프로젝트의 자산 경로는 config에서 오며, 머신별 차이는 §6의 `paths` 블록이 흡수한다. 환경변수는 §6.2의 보조 오버라이드로만 쓴다.

#### strict load 조건

`features_only=True`로 만든 timm 추출기는 classifier head를 갖지 않는다. 따라서 torchvision `resnet18-f37072fd.pth`를 그대로 넣으면 `fc.weight`/`fc.bias`가 unexpected key가 되어 `strict=True`는 실패한다. defectvad는 이를 `strict=False`로 넘겼는데, 그러면 backbone 본체가 통째로 어긋나도 조용히 지나간다.

이 프로젝트는 다음 조건으로 대체한다. `load_local_weights(..., strict=False)`로 적재하되 반환된 `missing`/`unexpected`를 팩토리가 직접 판정한다(`load_local_weights`는 이제 `(model, missing, unexpected)`를 반환한다 — 기존 5개 호출부는 모두 반환값을 버리는 단문 호출이라 영향 없다).

- `missing`이 비어 있지 않으면 `LocalAssetError`로 실패시킨다. 추출기가 요구하는 파라미터가 채워지지 않은 것이므로 예외 없이 오류다.
- `unexpected`는 **팩토리가 `state_dict()` 최상위 키 이름 집합을 미리 계산해, 그 집합에 없는 이름의 키만 허용**한다. 그 외(추출기가 실제로 갖고 있는 최상위 이름인데도 unexpected로 나온) 키가 하나라도 있으면 실패시킨다.
- 허용된 경우에도 `load_local_weights`가 무엇이 버려졌는지 로그로 남긴다.

**P2에서 확인된 사실 — classifier head뿐이 아니었다.** 당초 예상은 unexpected가 `fc.`/`head.` 접두사(분류기 head)뿐이라는 것이었다. 실제로 `layers=["layer1","layer2","layer3"]`(기본값)로 timm `features_only=True` 추출기를 만들면, timm이 `out_indices`에 없는 뒤쪽 스테이지(`layer4`)를 **모델에서 통째로 제거**한다. 따라서 torchvision 전체 resnet18 체크포인트를 얹으면 `layer4.*`도 `fc.*`와 함께 unexpected로 나온다(missing은 0개). 고정된 접두사 허용 목록(`fc.`, `head.`)은 이 경우를 놓친다. 대신 추출기가 실제로 갖고 있지 않은 최상위 서브모듈 이름의 키는 무엇이든 "구조적으로 존재하지 않아 버려도 되는 키"로 취급하고, 추출기가 갖고 있는 이름인데도 unexpected로 나오는 키만 실패로 판정한다 — 이쪽이 진짜 이름 불일치(값이 있어야 할 자리에 못 들어간 경우)를 가리키기 때문이다.

#### backbone 가중치 출처 불일치

`/mnt/d/backbones/resnet18-f37072fd.pth`는 torchvision ImageNet 가중치이고, anomalib이 쓰는 timm `resnet18`의 기본 pretrained는 별도 학습 레시피다. state_dict 키 이름(`conv1`, `bn1`, `layer1`~`layer4`, `fc`)은 겹칠 가능성이 높지만 **값이 다르므로 reference 성능 차이의 원인이 될 수 있다.**

defectvad가 같은 조합(timm `resnet18` + torchvision `.pth`)을 실사용했으므로 키 이름은 대체로 겹친다. 다만 defectvad는 `strict=False`만 걸어 두어 **실제로 몇 개의 키가 맞았는지는 코드상 확인되지 않는다.** 위 `strict load 조건`의 missing key 검사가 이 구멍을 막는다.

- P2에서 missing key 0개를 실제로 확인한다. 이름이 어긋나면 `key_map`으로 흡수하고, 흡수 불가능하면 필요한 가중치 파일과 경로를 사용자에게 알리고 대기한다(CON-004).
- P3에서 reference와 차이가 나면 PRD §5.4의 "pretrained weight" 항목으로 이 불일치를 먼저 검토한다.

## 5. 예상 변경 범위

| 모듈 | 변경 | 사유 |
|---|---|---|
| `tasks/anomaly/models/stfpm.py`, `models/efficientad.py` | 삭제 | §2 |
| `tasks/anomaly/models/custom_ae.py` | 유지 | v0.1 범위 밖. `models/` 디렉터리도 이 파일 때문에 존치(§3) |
| `tasks/anomaly/upstream/` | 신규 | anomalib 원본 복사 |
| `tasks/anomaly/adapters/` | 신규 | 모델별 adapter, 모델 생성 팩토리(§4.6) |
| `tasks/anomaly/adapter.py` | 수정 | 공통 `AnomalyAdapter`에서 `model.train_step` 위임 제거 |
| `tasks/anomaly/models/__init__.py` | 수정 | 삭제된 `stfpm`, `efficientad` import 제거 (현재 3개를 모두 import) |
| `tasks/anomaly/__init__.py` | 수정 | 새 `upstream`, `adapters` 패키지 등록 |
| `core/builders.py` | 추가 | `adam` 빌더 등록 |
| `core/engine.py` | 변경 없음 목표 | auxiliary loader는 adapter가 조달하므로 시그니처 유지(§4.5) |
| `core/adapter.py` | 변경 없음 목표 | 기존 hook으로 충분한지 P1에서 판단 |
| `configs/anomaly/stfpm.yaml`, `efficientad.yaml` | 수정 | optimizer/scheduler/경로, adapter params(auxiliary) |
| `configs/anomaly/_base.yaml` | 검토 | `batch_size: 8`·`epochs: 5`는 현재 값. EfficientAD는 batch size 1 필요(§4.5) |
| `core/config.py` | 수정 | `paths` 블록 해석과 `${paths.*}` 치환 (§6). 등급 C |
| `bench/runner.py` | 수정 | `resolve_split_configs`도 같은 치환을 거치게 한다 (§6.4) |
| `cli/commands.py`, `cli/parser.py` | 수정 | `check_assets` 치환, `--local-config` 인자 (§6.4) |
| `core/offline.py` | 수정 | `TORCH_HOME`·`HF_HOME` 기본값의 절대 경로 제거 (§6) |
| `configs/assets.yaml`, `configs/benchmarks/*.yaml` | 수정 | 절대 경로를 placeholder로 (§6.1) |
| `configs/local.example.yaml`, `.gitignore` | 신규·수정 | 머신별 `configs/local.yaml` 템플릿과 제외 규칙 (§6.2) |

`core/*` 변경 시 모델명 기반 분기를 두지 않는다(NFR-005). `MODELS` 등록 대상은 upstream `nn.Module` 자체가 아니라 `adapters/`의 팩토리 함수다 — 로컬 가중치 주입이 생성 시점에 필요하기 때문이며, 팩토리가 반환하는 것은 upstream 원본 `nn.Module`이다(§4.6).

## 6. 로컬 환경 경로 이식성

이 저장소는 WSL 워크스페이스에서 개발되고 GitHub와 동기화되지만, 이후 설비 등 여러 로컬 환경에서 **압축 파일로 받아 풀어서** 사용된다. 그 환경들의 dataset·backbone 경로는 개발 머신의 `/mnt/d/datasets`, `/mnt/d/backbones`와 다르다.

현재 cv_boilerplate는 절대 경로를 config에 직접 적는다. `configs/assets.yaml`, `configs/anomaly/_base.yaml`, `configs/anomaly/stfpm.yaml`, `configs/anomaly/efficientad.yaml`, `configs/benchmarks/` 3개 파일에 `/mnt/d/...`가 있고, `core/offline.py`의 `TORCH_HOME`·`HF_HOME` 기본값도 마찬가지다. 새 머신에서는 이 파일들을 일일이 고쳐야 하고, 고친 내용은 다음 zip을 받으면 사라진다.

목표는 **저장소 파일을 고치지 않고 머신마다 경로 두 개만 지정하면 동작**하는 것이다. 실제 사용된 절대 경로는 run 기록에 남아야 하고(NFR-002), 경로가 없으면 조용히 넘어가지 않아야 한다.

### 6.1 경로 루트 두 개

config에 최상위 `paths` 블록을 신설한다. `TOP_LEVEL_KEYS`에 `paths`를 추가한다.

```yaml
paths:
  dataset_root: /mnt/d/datasets      # 개발 머신 기본값. _base.yaml 에 기재
  backbone_root: /mnt/d/backbones
```

나머지 config는 절대 경로 대신 placeholder를 쓴다.

```yaml
data:
  root: ${paths.dataset_root}/mvtec
model:
  params:
    weights_path: ${paths.backbone_root}/resnet18-f37072fd.pth
```

### 6.2 해석 우선순위

높은 것이 이긴다.

| 순위 | 출처 | 용도 |
|---|---|---|
| 1 | CLI `--set paths.dataset_root=...` | 일회성 실험. 기존 override 메커니즘을 그대로 쓴다 |
| 2 | 환경변수 `DATASET_DIR`, `BACKBONE_DIR` | 세션 단위 오버라이드. 레거시 defectvad와 같은 이름이다 |
| 3 | `configs/local.yaml` | **머신별 SSOT.** `.gitignore`에 넣어 저장소와 zip에 포함되지 않는다 |
| 4 | config의 `paths` 블록 | `_base.yaml`의 개발 머신 기본값 |

`configs/local.yaml`은 `paths` 블록만 갖는 최소 파일이며, 배포본에는 `configs/local.example.yaml` 템플릿만 들어간다.

`BACKBONE_DIR` 환경변수를 **유일한** 경로 조달 수단으로 쓰는 defectvad 방식은 채택하지 않는다. 값이 run 기록에 남지 않아 재현이 보장되지 않기 때문이다(§4.6). 다만 위 2순위의 보조 수단으로는 채택한다.

### 6.3 치환과 실패 처리

`core/config.py`에 두 함수를 추가한다. 새 의존성은 도입하지 않는다(omegaconf·python-dotenv 모두 불필요).

- `resolve_paths(config, override_args=None, local_config_path=None)` — 위 우선순위로 `config["paths"]`를 확정하고, 무엇이 어느 출처에서 왔는지 `config["paths"]["_source"]`에 기록한다.
- `interpolate(config)` — merge·override가 끝난 config를 재귀 순회하며 문자열 안의 `${paths.<key>}`를 치환한다. 정의되지 않은 key를 만나면 사용 가능한 key를 나열한 `ConfigError`를 던진다.

**P1-T06 구현 중 추가 결정(P1 적대적 검증 A1에서 확인):** `_base.yaml`의 `paths` 블록이 `dataset_root`·`backbone_root`를 함께 선언해도, 개별 config가 `${paths.<key>}`로 실제로 참조하지 않는 키는 확정·검증 대상에서 제외한다(`used_placeholder_keys()`). 처음부터 학습하는 `custom_*` 모델처럼 `weights_path`가 없는 config까지 존재하지도 않는 `backbone_root`를 요구하면 NFR-003(공통 코드가 다른 4개 task를 불필요하게 막지 않아야 함)을 해친다. 선언되었지만 참조되지 않는 키는 `config["paths"]`에도 남기지 않는다.

확정된(=실제로 참조되는) 각 root에 대해 먼저 문자열 타입인지 확인하고(`--set paths.*=null` 등 비문자열 값을 걸러낸다), 그다음 `os.path.isdir` 검사를 한다. 없으면 `ConfigError`로 즉시 멈춘다. 메시지에는 어떤 root가 없는지, `configs/local.yaml`에 무엇을 적어야 하는지, 대응 환경변수 이름을 함께 적는다. **개발 머신 기본값으로 조용히 폴백하지 않는다**(CON-004와 같은 기조) — rank 4로 선택되었더라도 그 사실이 `_source`에 남고 부재 시 동일하게 실패하므로 "조용한" 폴백이 아니다.

파일 단위 검증은 기존 `core/offline.py#load_local_weights`가 `LocalAssetError`로 처리하므로 건드리지 않는다. 새 검사는 root 수준에서만 한다.

### 6.4 적용 지점

config가 만들어지는 경로는 두 곳이며 **둘 다** 치환을 거쳐야 한다.

- `core/config.py#resolve_config` — 일반 실행 경로. `cli/commands.py`가 호출한다.
- `bench/runner.py#resolve_split_configs` — benchmark 경로. `resolve_config`를 쓰지 않고 `load_and_merge_base` + `deep_merge` + `apply_overrides`를 직접 조합하므로 누락되기 쉽다.

치환은 `--set` 적용 **후**에 한다. `--set paths.backbone_root=...`가 placeholder에 반영되어야 하기 때문이다.

기존 `config.resolved.yaml` 저장은 치환 후 config를 저장하므로, 실제 사용된 절대 경로가 자동으로 run 기록에 남는다.

`cli/commands.py#check_assets`는 `configs/assets.yaml`을 직접 읽으므로 같은 치환을 거치게 하고, `--local-config` 인자를 `cli/parser.py`에 추가한다. `configs/assets.yaml` 자체도 최상위 `paths` 블록(개발 머신 기본값)을 갖는다. **P1 적대적 검증 A1에서 추가:** `--set`도 `check-assets`에 추가해 `apply_overrides` → `resolve_paths` → `interpolate` 순으로 호출한다 — 그렇지 않으면 이 커맨드만 §6.2의 1순위(CLI `--set`)를 쓸 수 없다. 이 명령이 새 머신의 첫 실행 절차가 된다 — zip 해제 → `configs/local.yaml` 작성 → `check-assets`로 누락 자산 확인.

### 6.5 범위와 등급

`core/config.py`는 5개 task가 공유하므로 이 변경은 공통 코드 변경 등급 C(계약 변경)다. 사용자 승인을 받았다(2026-08-20). `configs/` 하위 5개 task의 config에 남은 같은 패턴의 절대 경로도 함께 바꾼다.

이 작업은 P2·P4의 config 작성보다 **먼저** 끝나야 한다. 나중에 하면 STFPM·EfficientAD config를 두 번 쓰게 된다.

## 7. 미결정

| 항목 | 결정 시점 |
|---|---|
| ~~anomalib 대상 commit 고정~~ | 완료 — `v2.3.0` / `091ca6a` |
| ~~`upstream/components/`에 복사할 정확한 파일 목록~~ | 완료 — P0-T03, §3 트리 참조 |
| ~~기존 hook만으로 EfficientAD lifecycle이 충분한지~~ | 완료 — P5. 충분하지 않았다. task-agnostic hook `on_validation_start`를 `core/adapter.py`·`core/engine.py`에 추가(기본 no-op, §4.3) |
| ~~auxiliary transform이 `transform.py`에 별도로 필요한지 (§4.5)~~ | 완료 — P4. 불필요. auxiliary 전용 파이프라인은 `EfficientAdAdapter._build_auxiliary_loader`에 두고, 공통 `anomaly_default`에는 `normalize` 플래그만 추가 |
| ~~`InferenceBatch` 조달 방식~~ | 완료 — P0-T03, `dataclasses/torch/base.py` + `dataclasses/generic.py` 파일째 복사(§3) |
| ~~torchvision resnet18 state_dict의 unexpected key가 classifier head뿐인지 (§4.6)~~ | 완료 — P2. 아니었다: `layer4.*`도 unexpected(timm이 뒤쪽 스테이지를 제거). 최상위 서브모듈 존재 여부 기반 판정으로 대체(§4.6) |
| ~~EfficientAD 학습 budget — batch size 1에서 몇 epoch을 돌릴지~~ | 완료 — P4-T04. bottle(209 train, batch_size=1) 1 epoch ≈ 30초 측정, `train.epochs: 20`(≈10분) / `scheduler.step_size: 19`로 확정(`configs/anomaly/efficientad.yaml`) |
| ~~MVTec 대표 3개 카테고리 선정~~ | 완료 — P3-T01. bottle, carpet, capsule |
| 복수 parameter group을 요구하는 모델의 optimizer 표현 (§4.2) | v0.1 범위 밖 — v0.3 이월([RETROSPECTIVE.md](reports/RETROSPECTIVE.md) §4) |

P6 종료 시점의 미결정 항목 처리와 v0.2 이월 목록은 [RETROSPECTIVE.md](reports/RETROSPECTIVE.md)에 있다. `upstream/` 파일별 출처·라이선스는 [UPSTREAM-INVENTORY.md](reports/UPSTREAM-INVENTORY.md), 새 모델 추가 절차는 [MODEL-ADD.md](reports/MODEL-ADD.md)를 참조한다.

---

작성일: 2026-08-20
문서 상태: Reviewed — cv_boilerplate@65d5412 코드 대조 완료
