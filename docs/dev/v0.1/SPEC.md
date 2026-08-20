# SPEC — Anomaly Detection Integration on `cv_boilerplate`

상위 문서: [BRIEF.md](BRIEF.md) · [PRD.md](PRD.md) · 하위 문서: [PLAN.md](PLAN.md)

분석 기준: `cv_boilerplate@65d5412` (로컬: `../../../260818_cv-boilerplate`)

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
| forward 호출과 출력 변환 | 모델별 adapter | anomalib 출력 → `{"pred_score", "anomaly_map"}` |
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

다만 서로 다른 lr을 갖는 복수 parameter group을 요구하는 모델은 이 방식으로 표현할 수 없다. v0.1 범위 밖이므로 지금은 다루지 않고, 그런 모델을 만나면 `core/builders.py`에 group 지정 방식을 추가한다(§6).

### 4.3 모델별 adapter 구조

공통 `AnomalyAdapter`(metric·threshold·visualize)를 상속하고, 모델별 차이만 override한다.

```text
AnomalyAdapter                 # 기존: metric, threshold, smooth, visualize
├── StfpmAdapter               # train_step에서 upstream loss 호출
└── EfficientAdAdapter         # + on_fit_start: teacher 통계, auxiliary loader
                               # + on_fit_end: 분위수 calibration
```

#### `on_fit_end` 순서 제약

기존 `AnomalyAdapter.on_fit_end`는 두 가지를 순서대로 수행한다.

1. 모델의 `on_fit_end` 호출 (모델별 calibration)
2. `compute_thresholds(model, loaders["valid"], device, smooth_sigma)` — valid 전용 threshold 결정

`EfficientAdAdapter`가 `on_fit_end`를 override할 때 **반드시 `super().on_fit_end()`를 호출**하고, 분위수 calibration을 threshold 계산보다 **먼저** 끝내야 한다. 분위수는 `forward`가 두 anomaly map을 합성하는 스케일을 바꾸므로, calibration 전에 계산한 threshold는 calibration 후의 score와 비교 대상이 아니다.

`core/engine.py`는 `on_fit_end` 직전에 best checkpoint 가중치를 다시 로드한다. calibration이 best 가중치 기준으로 수행되도록 하기 위한 것이므로, 이 순서에 의존하는 hook은 `on_epoch_end`가 아니라 `on_fit_end`에 둔다.

### 4.4 STFPM

| 항목 | 값 | 위치 |
|---|---|---|
| optimizer | SGD, lr 0.4, momentum 0.9, weight_decay 0.001 | `configs/` |
| scheduler | 없음 | `configs/` |
| backbone | `/mnt/d/backbones/resnet18-f37072fd.pth` | `configs/` |
| loss | upstream `loss.py` 호출 | `StfpmAdapter.train_step` |
| teacher freeze | 항상 eval() 유지 | adapter 또는 모델 생성 시 |

### 4.5 EfficientAD

| 항목 | 값 | 위치 |
|---|---|---|
| optimizer | Adam, lr 1e-4, weight_decay 1e-5 | `configs/` (`adam` 빌더 추가 필요) |
| scheduler | StepLR, 95% 시점 0.1배 | `configs/` (기존 `step` 빌더 사용) — 아래 참조 |
| pretrained teacher | `/mnt/d/backbones/efficientad_pretrained_weights/` | `configs/` |
| auxiliary 데이터 | `/mnt/d/datasets/imagenette2` | `configs/`(경로) + `EfficientAdAdapter`(loader 생성·소비) — 아래 참조 |
| batch size 1, normalization 미사용 | 제약 | `configs/` |

#### StepLR `step_size` 산출

`build_step_scheduler(target, step_size, gamma=0.1)`는 **절대 epoch 수**를 받는다. "전체 학습의 95% 시점"은 `train.epochs`에서 파생되는 값이므로, config에 상수로 적으면 `epochs`를 바꿀 때 조용히 어긋난다.

v0.1은 `step_size`를 config에 직접 기입하고, 해당 config의 `train.epochs`와 짝이 맞는지 주석으로 남긴다. 파생 값을 config에서 표현하는 일반적 방법(`epochs` 참조 문법 등)은 도입하지 않는다 — 한 모델만 요구하는 기능이므로 미룬다 — 두 모델 이상에서 같은 필요가 확인되기 전에는 새 추상화를 추가하지 않는다.

#### auxiliary 데이터 조달 경로

`core/engine.py#Trainer.fit`은 `train_loader`/`valid_loader`만 받고, `adapter.on_fit_start(model, loaders, device)`의 `loaders`도 `{"train", "valid"}`뿐이다. ImageNette auxiliary 스트림은 이 경로로 들어오지 않는다.

**`EfficientAdAdapter`가 스스로 DataLoader를 만든다.** `on_fit_start`에서 config의 auxiliary 경로를 읽어 `core/builders.py#build_dataloader`로 loader를 생성하고, adapter 인스턴스에 보관한 뒤 `train_step`에서 배치를 하나씩 꺼내 쓴다. train_loader보다 짧으면 순환한다.

`core/engine.py`의 시그니처는 바꾸지 않는다. auxiliary 스트림은 EfficientAD 하나만 요구하는 것이므로, 공통 engine에 세 번째 loader 개념을 추가하면 나머지 4개 task가 쓰지 않는 인자를 떠안게 된다(NFR-003·NFR-005).

- adapter가 auxiliary 경로를 읽으려면 adapter params로 전달되어야 한다 → `configs/anomaly/efficientad.yaml`의 `adapter.params`에 둔다.
- auxiliary transform(normalization 미사용 제약 포함)이 `tasks/anomaly/transform.py`에 추가로 필요한지는 upstream `torch_model.py` 확인 후 P4에서 판단한다(§6).

Lightning hook 매핑:

| anomalib `lightning_model.py` | 이 프로젝트 |
|---|---|
| `on_train_start` — teacher 채널 통계, auxiliary 준비 | `EfficientAdAdapter.on_fit_start` |
| `on_validation_start` — 90%/99.5% 분위수 | `on_fit_end` (또는 `on_epoch_end`) |
| `configure_optimizers` | `configs/` + `core/builders.py` |

Adam 빌더는 현재 registry에 없다(`adamw`, `sgd`만 존재). `core/builders.py`에 `adam`을 추가한다 — 모델명 분기가 아닌 범용 빌더이므로 NFR-005에 저촉되지 않는다.

## 5. 예상 변경 범위

| 모듈 | 변경 | 사유 |
|---|---|---|
| `tasks/anomaly/models/stfpm.py`, `models/efficientad.py` | 삭제 | §2 |
| `tasks/anomaly/models/custom_ae.py` | 유지 | v0.1 범위 밖. `models/` 디렉터리도 이 파일 때문에 존치(§3) |
| `tasks/anomaly/upstream/` | 신규 | anomalib 원본 복사 |
| `tasks/anomaly/adapters/` | 신규 | 모델별 adapter |
| `tasks/anomaly/adapter.py` | 수정 | 공통 `AnomalyAdapter`에서 `model.train_step` 위임 제거 |
| `tasks/anomaly/models/__init__.py` | 수정 | 삭제된 `stfpm`, `efficientad` import 제거 (현재 3개를 모두 import) |
| `tasks/anomaly/__init__.py` | 수정 | 새 `upstream`, `adapters` 패키지 등록 |
| `core/builders.py` | 추가 | `adam` 빌더 등록 |
| `core/engine.py` | 변경 없음 목표 | auxiliary loader는 adapter가 조달하므로 시그니처 유지(§4.5) |
| `core/adapter.py` | 변경 없음 목표 | 기존 hook으로 충분한지 P1에서 판단 |
| `configs/anomaly/stfpm.yaml`, `efficientad.yaml` | 수정 | optimizer/scheduler/경로, adapter params(auxiliary) |
| `configs/anomaly/_base.yaml` | 검토 | `batch_size: 8`·`epochs: 5`는 현재 값. EfficientAD는 batch size 1 필요(§4.5) |

`core/*` 변경 시 모델명 기반 분기를 두지 않는다(NFR-005).

## 6. 미결정

| 항목 | 결정 시점 |
|---|---|
| anomalib 대상 commit 고정 | P0 |
| `upstream/components/`에 복사할 정확한 파일 목록 | P0 (실제 import 추적 후) |
| 기존 hook만으로 EfficientAD lifecycle이 충분한지 | P1 |
| auxiliary transform이 `transform.py`에 별도로 필요한지 (§4.5) | P4 (upstream `torch_model.py` 확인 후) |
| EfficientAD 학습 budget — batch size 1에서 몇 epoch을 돌릴지 | P4 (PRD §5에 따라 사용자 실행 시간과 함께 판단) |
| MVTec 대표 3개 카테고리 선정 | P3 |
| 복수 parameter group을 요구하는 모델의 optimizer 표현 (§4.2) | v0.1 범위 밖 |

---

작성일: 2026-08-20
문서 상태: Reviewed — cv_boilerplate@65d5412 코드 대조 완료
