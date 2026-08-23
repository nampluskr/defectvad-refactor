# UPSTREAM-INVENTORY (v0.2) — 복사 원본 출처·라이선스 인벤토리

선행 문서: [v0.1 UPSTREAM-INVENTORY](../../v0.1/reports/UPSTREAM-INVENTORY.md) · 절차: [MODEL-ADD](../../v0.1/reports/MODEL-ADD.md)

v0.2에서 추가된 anomalib 복사본을 기록한다. 대상은 **FastFlow**와 **PatchCore**이며, STFPM·EfficientAD 행은 v0.1 문서가 SSOT이라 여기서 반복하지 않는다.

## 1. v0.1 문서와의 차이

| 항목 | v0.1 | v0.2 |
|---|---|---|
| 복사본 루트 | `src/tasks/anomaly/upstream/` | `src/tasks/anomaly/models/` |
| 라이선스 | Apache-2.0 단일 | Apache-2.0 + **MIT** (§3) |
| 외부 런타임 의존성 | timm | timm, **FrEIA**, **kornia**, **scikit-learn**, tqdm (§5) |

v0.1 문서 §2~§5의 경로 표기는 `upstream/` 기준이라 현재 트리와 맞지 않는다. 파일 내용·해시는 유효하다.

## 2. 출처

| 항목 | 값 |
|---|---|
| 저장소 | `anomalib` (open-edge-platform) |
| commit | `091ca6aca92c8d0e416394f79e52f5a3cea3db73` (`v2.3.0`) — v0.1과 동일 핀 |
| 로컬 경로 | `/mnt/d/projects/clones/anomalib` |

v0.2에서 sparse-checkout에 추가한 경로는 다음과 같다. 모두 핀을 바꾸지 않고 해당 디렉터리만 추가로 받은 것이다.

| 경로 | 사유 |
|---|---|
| `src/anomalib/models/image/fastflow` | FastFlow 원본 |
| `src/anomalib/models/image/patchcore` | PatchCore 원본 |
| `src/anomalib/utils` | PatchCore `torch_model.py`의 `@deprecate` 데코레이터 |

상류 최신 태그는 `v2.3.0`으로 핀과 동일하다. 핀 상향 사유 없음.

## 3. 라이선스 — v0.2에서 MIT가 유입된다

헤더는 원본 그대로 보존한다. 지우거나 재작성하는 것은 CON-001 위반이자 라이선스 위반이다.

| 파일 | 라이선스 | 저작권 |
|---|---|---|
| `fastflow/torch_model.py` | Apache-2.0 | `(c) 2022 @gathierry` (원저자) + `(C) 2022-2025 Intel Corporation` (수정) |
| `fastflow/loss.py`, `fastflow/anomaly_map.py` | Apache-2.0 | `(C) 2022-2025 Intel Corporation` |
| `components/flow/all_in_one_block.py` | **MIT** | `(c) https://github.com/vislearn/FrEIA` |
| `patchcore/*.py` | Apache-2.0 | `(C) 2022-2025 Intel Corporation` |
| `components/{base,sampling,dimensionality_reduction,filters,utils}/*.py` | Apache-2.0 | `(C) 2022-2025 Intel Corporation` |

`all_in_one_block.py`는 anomalib이 FrEIA에서 벤더링한 파일이라 MIT다. v0.1 §1의 "복사한 파일 모두 Apache-2.0" 서술은 v0.2부터 성립하지 않는다. 배포 시 MIT 고지도 함께 포함해야 한다. PatchCore가 가져온 7개 파일은 전부 Apache-2.0이라 추가 라이선스는 없다.

## 4. 파일 인벤토리

대상 경로는 `src/tasks/anomaly/models/` 기준, 원본은 `/mnt/d/projects/clones/anomalib/` 기준이다.

### 4.1 FastFlow

| 대상 | 원본 | 줄 수 | sha256 | 허용 변경 |
|---|---|---|---|---|
| `fastflow/torch_model.py` | `src/anomalib/models/image/fastflow/torch_model.py` | 273 | `a7b57151b46e26ff` | import 2건 |
| `fastflow/loss.py` | `src/anomalib/models/image/fastflow/loss.py` | 65 | `9c56354ae64a7060` | 없음 |
| `fastflow/anomaly_map.py` | `src/anomalib/models/image/fastflow/anomaly_map.py` | 84 | `d1f9eadefc9502d2` | 없음 |
| `components/flow/all_in_one_block.py` | `src/anomalib/models/components/flow/all_in_one_block.py` | 425 | `2b928f1fdfe9b82f` | 없음 |

### 4.2 PatchCore

| 대상 | 원본 | 줄 수 | sha256 | 허용 변경 |
|---|---|---|---|---|
| `patchcore/torch_model.py` | `src/anomalib/models/image/patchcore/torch_model.py` | 444 | `a8a2422279bc95d0` | import 4건 |
| `patchcore/anomaly_map.py` | `src/anomalib/models/image/patchcore/anomaly_map.py` | 111 | `800c333d4066ef59` | import 1건 |
| `components/base/dynamic_buffer.py` | `src/anomalib/models/components/base/dynamic_buffer.py` | 90 | `ef214c097f6e674c` | 없음 |
| `components/sampling/k_center_greedy.py` | `src/anomalib/models/components/sampling/k_center_greedy.py` | 148 | `89d265ab66264b18` | import 1건 |
| `components/dimensionality_reduction/random_projection.py` | `src/anomalib/models/components/dimensionality_reduction/random_projection.py` | 196 | `65a2794fa72348ef` | 없음 |
| `components/filters/blur.py` | `src/anomalib/models/components/filters/blur.py` | 141 | `337adeedbb5e0ba4` | 없음 |
| `components/utils/deprecation.py` | `src/anomalib/utils/deprecation.py` | 338 | `720ccb0ea6aa1a2e` | 없음 |

`__init__.py`와 `lightning_model.py`는 두 모델 모두 복사하지 않았다. 전자는 패키지 `__init__`이 Lightning 경로를 끌어오기 때문이고, 후자는 CON-002 때문이다. `components/` 아래에도 `__init__.py`를 두지 않는다(기존 관례대로 namespace package).

### 4.3 허용된 변경 — import 경로 전량

CON-001이 허용하는 유일한 변경이다. 아래가 전부이며, 그 외 차이는 11개 파일 모두 0라인이다.

| 파일 | 변경 후 |
|---|---|
| `fastflow/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `fastflow/torch_model.py` | `...components.flow.all_in_one_block import AllInOneBlock` |
| `patchcore/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `patchcore/torch_model.py` | `...components.base.dynamic_buffer import DynamicBufferMixin` |
| `patchcore/torch_model.py` | `...components.sampling.k_center_greedy import KCenterGreedy` |
| `patchcore/torch_model.py` | `...components.feature_extractors.timm import TimmFeatureExtractor` |
| `patchcore/torch_model.py` | `...components.utils.deprecation import deprecate` |
| `patchcore/torch_model.py` | `...components.tiler import Tiler` (`TYPE_CHECKING` 블록) |
| `patchcore/anomaly_map.py` | `...components.filters.blur import GaussianBlur2d` |
| `components/sampling/k_center_greedy.py` | `...dimensionality_reduction.random_projection import SparseRandomProjection` |

`patchcore/torch_model.py`의 원본 한 줄 `from anomalib.models.components import DynamicBufferMixin, KCenterGreedy, TimmFeatureExtractor`는 패키지 `__init__`을 경유하면 Lightning이 딸려 오므로 **모듈 단위 3줄로 분리**했다. 줄 수가 늘지만 성격은 import 경로 치환이다.

### 4.4 무결성 확인

```bash
cd src/tasks/anomaly/models && sha256sum \
  fastflow/*.py patchcore/*.py \
  components/flow/all_in_one_block.py components/base/dynamic_buffer.py \
  components/sampling/k_center_greedy.py \
  components/dimensionality_reduction/random_projection.py \
  components/filters/blur.py components/utils/deprecation.py
```

## 5. 신규 런타임 의존성

| 패키지 | 상태 (`pytorch_env`) | 사용처 | `requirements.txt` |
|---|---|---|---|
| `FrEIA` | 설치됨 | FastFlow `SequenceINN`, `all_in_one_block` | **미기재** |
| `kornia` | 0.8.2 | PatchCore `filters/blur.py` (`GaussianBlur2d`) | **미기재** |
| `scikit-learn` | 1.7.2 | PatchCore `random_projection.py` | **미기재** |
| `scipy` | 1.15.3 | `all_in_one_block.py` | **미기재** |
| `timm` | 1.0.22 | 공통 backbone | **미기재** |
| `omegaconf` | 2.3.0 | FastFlow `anomaly_map.py` 타입 힌트 | **미기재** |
| `tqdm` | 4.67.1 | `k_center_greedy.py` | 기재됨(간접) |

어느 것도 런타임에 네트워크를 쓰지 않는다(CON-003 무관). 다만 `requirements.txt`에 없어 새 머신에서 재현이 깨진다 — v0.2 범위에서 보강 대상이다.

## 6. 모델 연결 — FastFlow

| 항목 | 값 |
|---|---|
| upstream 파일 | `fastflow/torch_model.py`, `fastflow/loss.py`, `fastflow/anomaly_map.py` |
| 공유 components | `components/flow/all_in_one_block.py` (신규), `data/torch_base.py`, `data/generic.py` |
| adapter | `adapters/fastflow.py#FastflowAdapter` (`ADAPTERS: "fastflow"`) |
| 모델 팩토리 | `models/fastflow/__init__.py#build_fastflow` (`MODELS: "fastflow"`, `"fastflow_anomaly"`) |
| config | `configs/anomaly/models/fastflow.yaml` |
| override한 hook | `train_step`만 |
| 로컬 자산 | `${paths.backbone_root}/resnet18-f37072fd.pth` 또는 `wide_resnet50_2-95faca4d.pth` |
| 자산 주입 지점 | 팩토리. `pre_trained=False`로 생성 후 `feature_extractor`에 로컬 가중치 로드 |

### 6.1 `lightning_model.py` 이관 결과

| anomalib | 이 프로젝트 |
|---|---|
| `configure_optimizers` → `Adam(lr=0.001, weight_decay=0.00001)`, scheduler 없음 | config `optim` |
| `training_step` → `FastflowLoss(hidden_variables, jacobians)` | `FastflowAdapter.train_step` |
| `validation_step` → `InferenceBatch` 반환 | 공통 `AnomalyAdapter.eval_step` (override 불필요) |
| `trainer_arguments` → `gradient_clip_val=0` | config `train.grad_clip: null` |
| `input_size` 필수 | `model.params.input_size` (§6.2) |
| `pre_trained=True` | 팩토리에서 `False` + 로컬 가중치 |

lifecycle hook이 하나도 필요 없다. 백본은 생성자에서 freeze되고, `torch_model.forward`가 매 호출 `feature_extractor.eval()`을 강제하므로 engine의 `model.train()`이 백본을 학습 모드로 되돌리지 못한다.

### 6.2 `input_size` 중복 — 알려진 제약

`build_model`(`src/core/builders.py:57`)은 `config["model"]`만 받으므로 `data.image_size`에 접근할 수 없다. FastFlow는 LayerNorm과 flow block을 고정 공간 크기로 만들기 때문에 `input_size`가 생성 시점에 필요하다. 따라서 `model.params.input_size`에 상수로 적고 `data.image_size`와 짝임을 주석으로 명시했다.

두 값이 어긋나면 LayerNorm에서 shape 오류로 **즉시 실패**한다. 근본 해결은 `build_model`이 `image_size`를 받도록 공통 코드를 바꾸는 것이며 v0.2 범위 밖이다.

## 7. 모델 연결 — PatchCore

| 항목 | 값 |
|---|---|
| upstream 파일 | `patchcore/torch_model.py`, `patchcore/anomaly_map.py` |
| 공유 components | `base/dynamic_buffer.py`, `sampling/k_center_greedy.py`, `dimensionality_reduction/random_projection.py`, `filters/blur.py`, `utils/deprecation.py` (전부 신규), `feature_extractors/timm.py`, `data/torch_base.py`, `data/generic.py` |
| adapter | `adapters/patchcore.py#PatchcoreAdapter` (`ADAPTERS: "patchcore"`) |
| 모델 팩토리 | `models/patchcore/__init__.py#build_patchcore` (`MODELS: "patchcore"`, `"patchcore_anomaly"`) |
| config | `configs/anomaly/models/patchcore.yaml` |
| override한 hook | `train_step`, `on_validation_start` |
| 로컬 자산 | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` (기본) 또는 `resnet18-f37072fd.pth` |
| 자산 주입 지점 | 팩토리. `pre_trained=False`로 생성 후 `feature_extractor.feature_extractor`(내부 timm 모듈)에 로드 |

### 7.1 `lightning_model.py` 이관 결과

| anomalib | 이 프로젝트 |
|---|---|
| `configure_optimizers` → **None** (최적화 없음) | config `optim`에 no-op optimizer 선언 (§7.2) |
| `training_step` → embedding 수집 + `torch.tensor(0.0, requires_grad=True)` | `PatchcoreAdapter.train_step` (동일한 dummy loss) |
| `MemoryBankMixin.on_validation_start` → `fit()` → `subsample_embedding(ratio)` | `PatchcoreAdapter.on_validation_start` (§7.3) |
| `validation_step` → `InferenceBatch` 반환 | 공통 `AnomalyAdapter.eval_step` (override 불필요) |
| `trainer_arguments` → `max_epochs=1` | config `train.epochs: 1` |
| `trainer_arguments` → `gradient_clip_val=0` | config `train.grad_clip: null` |
| `precision` (float16/float32 전환) | 이관하지 않음. 공통 `runtime.amp`가 담당하며 기본 float32 |
| `configure_pre_processor`의 `center_crop_size` | 이관하지 않음. anomalib 기본값도 `None`이라 resize만 하는 현재 data config와 동치 |
| `learning_type` | 대응물 없음 |

### 7.2 optimizer — 의도적으로 no-op

PatchCore는 gradient 학습을 하지 않는다. 그러나 공통 engine은 optimizer를 필수로 요구하고(`engine.py:126,134`) 매 스텝 `loss.backward()`를 호출한다(`engine.py:130`). 이를 다음과 같이 충족했다.

- adapter가 anomalib과 동일하게 `torch.tensor(0.0, requires_grad=True)`를 반환한다. leaf tensor이므로 backward는 성공하되 어떤 파라미터에도 gradient를 만들지 않는다.
- 팩토리는 백본을 **freeze하지 않는다.** 전부 freeze하면 `build_optimizer`(`builders.py:27`)에 빈 파라미터 리스트가 넘어가 PyTorch가 거부한다. `TimmFeatureExtractor.forward`가 `torch.no_grad()` 안에서 실행되므로 gradient가 애초에 생기지 않아, freeze 없이도 백본은 실질적으로 고정된다.
- 실측으로 확인했다: `optimizer.step()` 후 전 파라미터의 `.grad`가 `None`이다.
- `runtime.amp`는 기본 `false`다. `true`로 켜면 `GradScaler`가 "No inf checks were recorded"로 실패할 수 있으므로 PatchCore에서는 켜지 않는다.

`TimmFeatureExtractor`가 `forward`마다 `eval()`을 강제하므로, EfficientAD식 인스턴스 수준 `train` 바인딩 고정은 불필요하다.

### 7.3 memory bank 구축 시점

`subsample_embedding`은 **첫 validation 직전**에 한 번만 실행해야 한다. 더 이르면 embedding이 덜 모였고, 더 늦으면 evaluate가 빈 memory bank로 `ValueError`를 낸다. `engine.py:52`가 validation 직전에 `adapter.on_validation_start`를 부르므로 여기가 정확한 자리이며, anomalib의 `MemoryBankMixin.on_validation_start`와 대응한다.

- `_memory_bank_fitted` 플래그로 1회만 수행한다. `epochs: 1`이 기본이라 실제로는 한 번만 불린다.
- `KCenterGreedy`와 `SparseRandomProjection`이 난수를 소비하므로 `torch.random.fork_rng`로 감쌌다. 감싸지 않으면 valid split 크기가 학습 RNG를 교란한다.
- `memory_bank`는 크기가 가변인 registered buffer다. checkpoint 재로드는 `DynamicBufferMixin._load_from_state_dict`가 처리하며, evaluate 스모크로 실제 복원을 확인했다.

### 7.4 upstream `@deprecate` 경고

`subsample_embedding`은 `@deprecate(args={"embeddings": None})`가 붙어 있는데, `_warn_deprecated_arguments`가 `bound_args.arguments` 존재만 검사해 **기본값이 채워진 경우에도** 경고를 낸다. 따라서 `embeddings`를 넘기지 않아도 `DeprecationWarning`이 1회 출력된다. upstream 동작이므로 수정하지 않는다(CON-001). 무해하다.

### 7.5 가중치 로드 판정

`_load_backbone_weights`는 FastFlow와 동일한 규칙이다.

- `missing`이 하나라도 있으면 `LocalAssetError`.
- `unexpected`는 extractor가 갖고 있지 않은 최상위 서브모듈만 허용한다. `wide_resnet50_2` + `layers=[layer2, layer3]`는 `out_indices=[2,3]`, `out_dims=[512,1024]`로 해석되며 `layer1`·`layer4`·`fc`가 제거된다.
- shape 불일치(`RuntimeError`)는 `LocalAssetError`로 감싼다. wide_resnet50_2 자리에 resnet18 가중치를 넣으면 거부되는 것을 확인했다.
- 조용한 랜덤 초기화 폴백은 없다(CON-004).

## 8. 검증 기록

| 항목 | FastFlow | PatchCore |
|---|---|---|
| 원본 diff | import 2줄 외 0라인 | import 6줄 외 0라인 |
| `grep -rn lightning` | 0건 | docstring `:class:` 참조 3건만 (실행 경로 없음) |
| `git status -- src/core/ scripts/` | 무변경 | 무변경 |
| 기존 모델 파일 | 무변경 | 무변경 |
| freeze/grad | extractor trainable 0개, 전체 3,510,528개 | 의도적 미freeze. `optimizer.step()` 후 전 파라미터 `.grad is None` |
| train 스모크 (bottle) | 1 epoch → image 0.995 / pixel 0.954 | 1 epoch → image 1.000 / pixel 0.985 |
| evaluate 스모크 (bottle, test) | image 0.976 / pixel 0.965 | image 1.000 / pixel 0.988 |
| predict 스모크 (20장) | 정상 | 정상 |
| checkpoint 재로드 | 정상 | 가변 buffer 복원 정상 (`DynamicBufferMixin`) |

`src/core/`와 `scripts/`를 건드리지 않았으므로 기존 모델 재스모크는 불필요하다.

## 9. 미완 항목

- FastFlow `train.epochs: 100`은 잠정값이다. 핀된 클론에 `examples/configs`가 sparse-checkout되어 있지 않아 anomalib의 공식 학습 예산을 확인하지 못했다.
- 3개 카테고리(bottle, carpet, capsule) 성능 검증 — 사용자 실행 대기.
- 반대 벤더 CLI 적대적 검증 — 미실행.
- `requirements.txt`에 `FrEIA`·`kornia`·`scikit-learn`·`scipy`·`timm`·`omegaconf` 누락 (§5).
- FastFlow의 cait/deit 백본 미지원. 로컬 자산이 HF safetensors 디렉터리라 현재 `torch.load` 경로로 읽히지 않는다.
- PatchCore는 `runtime.amp: true`와 호환되지 않을 수 있다 (§7.2). 검증하지 않았다.

---

작성일: 2026-08-23
문서 상태: FastFlow·PatchCore 추가 산출물 (anomalib `091ca6a` 기준)
