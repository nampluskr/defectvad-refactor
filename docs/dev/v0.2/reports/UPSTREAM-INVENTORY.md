# UPSTREAM-INVENTORY (v0.2) — 복사 원본 출처·라이선스 인벤토리

선행 문서: [v0.1 UPSTREAM-INVENTORY](../../v0.1/reports/UPSTREAM-INVENTORY.md) · 절차: [MODEL-ADD](../../v0.1/reports/MODEL-ADD.md)

v0.2에서 추가된 anomalib 복사본을 기록한다. 대상은 **FastFlow**, **PatchCore**, **PaDiM**, **Reverse Distillation**이며, STFPM·EfficientAD 행은 v0.1 문서가 SSOT이라 여기서 반복하지 않는다.

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
| `src/anomalib/models/image/padim` | PaDiM 원본 |
| `src/anomalib/models/image/reverse_distillation` | Reverse Distillation 원본 |

`src/anomalib/models/components`는 이미 포함되어 있어 PaDiM용 `stats/multi_variate_gaussian.py`와 Reverse Distillation용 `backbone/resnet_decoder.py`는 추가 fetch가 필요 없었다.

상류 최신 태그는 `v2.3.0`으로 핀과 동일하다. 핀 상향 사유 없음. PaDiM 추가 시점에 `ls-remote --tags`로 재확인했으며 최신 태그는 여전히 `v2.3.0`이다.

## 3. 라이선스 — v0.2에서 MIT가 유입된다

헤더는 원본 그대로 보존한다. 지우거나 재작성하는 것은 CON-001 위반이자 라이선스 위반이다.

| 파일 | 라이선스 | 저작권 |
|---|---|---|
| `fastflow/torch_model.py` | Apache-2.0 | `(c) 2022 @gathierry` (원저자) + `(C) 2022-2025 Intel Corporation` (수정) |
| `fastflow/loss.py`, `fastflow/anomaly_map.py` | Apache-2.0 | `(C) 2022-2025 Intel Corporation` |
| `components/flow/all_in_one_block.py` | **MIT** | `(c) https://github.com/vislearn/FrEIA` |
| `patchcore/*.py` | Apache-2.0 | `(C) 2022-2025 Intel Corporation` |
| `padim/*.py` | Apache-2.0 | `(C) 2022-2025 Intel Corporation` |
| `reverse_distillation/torch_model.py`, `reverse_distillation/loss.py` | Apache-2.0 | `(C) 2022-2025 Intel Corporation` |
| `reverse_distillation/anomaly_map.py`, `reverse_distillation/components/bottleneck.py` | **MIT** + Apache-2.0 (이중 헤더) | `(c) 2022 hq-deng` (원저자) + `(C) 2022-2025 Intel Corporation` (수정) |
| `components/backbone/resnet_decoder.py` | Apache-2.0 | `(C) 2025 Intel Corporation` |
| `components/{base,sampling,dimensionality_reduction,filters,stats,utils}/*.py` | Apache-2.0 | `(C) 2022-2025 Intel Corporation` |

Reverse Distillation은 MIT가 들어오는 **두 번째** 경로다. `anomaly_map.py`와 `components/bottleneck.py`는 hq-deng의 RD4AD 구현에 기반해 MIT·Apache-2.0 헤더를 둘 다 달고 있다. anomalib이 이 모델 디렉터리에 `LICENSE` 파일을 따로 두었으므로 그 파일도 `models/reverse_distillation/LICENSE`로 함께 복사해 귀속을 보존했다. 원본 파일이 아니라 라이선스 고지의 추가이므로 CON-001과 무관하다.

`all_in_one_block.py`는 anomalib이 FrEIA에서 벤더링한 파일이라 MIT다. v0.1 §1의 "복사한 파일 모두 Apache-2.0" 서술은 v0.2부터 성립하지 않는다. 배포 시 MIT 고지도 함께 포함해야 한다. PatchCore가 가져온 7개 파일과 PaDiM이 가져온 3개 파일은 전부 Apache-2.0이라 추가 라이선스는 없다.

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

### 4.3 PaDiM

| 대상 | 원본 | 줄 수 | sha256 | 허용 변경 |
|---|---|---|---|---|
| `padim/torch_model.py` | `src/anomalib/models/image/padim/torch_model.py` | 243 | `64de58a120dc7949` | import 4건 |
| `padim/anomaly_map.py` | `src/anomalib/models/image/padim/anomaly_map.py` | 193 | `8fe539208957ed3b` | import 1건 |
| `components/stats/multi_variate_gaussian.py` | `src/anomalib/models/components/stats/multi_variate_gaussian.py` | 180 | `5034c641d0873b08` | import 1건 |

PaDiM이 재사용한 기존 components는 `feature_extractors/timm.py`, `feature_extractors/utils.py`, `filters/blur.py`, `base/dynamic_buffer.py`, `data/torch_base.py`이며 신규 복사는 `stats/multi_variate_gaussian.py` 하나뿐이다.

`__init__.py`와 `lightning_model.py`는 세 모델 모두 복사하지 않았다. 전자는 패키지 `__init__`이 Lightning 경로를 끌어오기 때문이고, 후자는 CON-002 때문이다. `components/` 아래에도 `__init__.py`를 두지 않는다(기존 관례대로 namespace package).

### 4.4 Reverse Distillation

| 대상 | 원본 | 줄 수 | sha256 | 허용 변경 |
|---|---|---|---|---|
| `reverse_distillation/torch_model.py` | `src/anomalib/models/image/reverse_distillation/torch_model.py` | 168 | `455ca6ddb3489400` | import 5건 |
| `reverse_distillation/loss.py` | `src/anomalib/models/image/reverse_distillation/loss.py` | 90 | `4496070473a0ea0c` | 없음 |
| `reverse_distillation/anomaly_map.py` | `src/anomalib/models/image/reverse_distillation/anomaly_map.py` | 117 | `c594bb007a26d3c2` | import 1건 |
| `reverse_distillation/components/bottleneck.py` | `src/anomalib/models/image/reverse_distillation/components/bottleneck.py` | 239 | `174c24f3f77b5a36` | 없음 |
| `components/backbone/resnet_decoder.py` | `src/anomalib/models/components/backbone/resnet_decoder.py` | 513 | `5fc005e1acff3c3b` | 없음 |
| `reverse_distillation/LICENSE` | `src/anomalib/models/image/reverse_distillation/LICENSE` | — | — | 없음 (라이선스 고지, §3) |

재사용한 기존 components는 `feature_extractors/timm.py`, `filters/blur.py`, `data/torch_base.py`이며 신규 복사는 `backbone/resnet_decoder.py` 하나뿐이다. `resnet_decoder.py`는 `components/`가 이미 sparse-checkout에 있어 추가 fetch 없이 확보됐다.

모델 하위 패키지인 `reverse_distillation/components/__init__.py`는 복사하지 않았다. Lightning을 끌어오지는 않지만(`from .bottleneck import get_bottleneck_layer` 한 줄뿐), `components/` 아래에 `__init__.py`를 두지 않는 기존 관례를 유지하고 대신 `torch_model.py`의 import를 모듈 단위로 바꿨다(§4.5).

### 4.5 허용된 변경 — import 경로 전량

CON-001이 허용하는 유일한 변경이다. 아래가 전부이며, 그 외 차이는 19개 파일 모두 0라인이다.

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
| `padim/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `padim/torch_model.py` | `...components.stats.multi_variate_gaussian import MultiVariateGaussian` |
| `padim/torch_model.py` | `...components.feature_extractors.timm import TimmFeatureExtractor` |
| `padim/torch_model.py` | `...components.feature_extractors.utils import dryrun_find_featuremap_dims` |
| `padim/torch_model.py` | `...components.tiler import Tiler` (`TYPE_CHECKING` 블록) |
| `padim/anomaly_map.py` | `...components.filters.blur import GaussianBlur2d` |
| `components/stats/multi_variate_gaussian.py` | `...components.base.dynamic_buffer import DynamicBufferMixin` |
| `reverse_distillation/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `reverse_distillation/torch_model.py` | `...components.feature_extractors.timm import TimmFeatureExtractor` |
| `reverse_distillation/torch_model.py` | `...components.backbone.resnet_decoder import get_decoder` |
| `reverse_distillation/torch_model.py` | `from .components.bottleneck import get_bottleneck_layer` (하위 패키지 `__init__` 우회) |
| `reverse_distillation/torch_model.py` | `...components.tiler import Tiler` (`TYPE_CHECKING` 블록) |
| `reverse_distillation/anomaly_map.py` | `...components.filters.blur import GaussianBlur2d` |

`patchcore/torch_model.py`의 원본 한 줄 `from anomalib.models.components import DynamicBufferMixin, KCenterGreedy, TimmFeatureExtractor`는 패키지 `__init__`을 경유하면 Lightning이 딸려 오므로 **모듈 단위 3줄로 분리**했다. `padim/torch_model.py`의 `from anomalib.models.components import MultiVariateGaussian, TimmFeatureExtractor`도 같은 이유로 2줄로 분리했다. 줄 수가 늘지만 성격은 import 경로 치환이다.

### 4.6 무결성 확인

```bash
cd src/tasks/anomaly/models && sha256sum \
  fastflow/torch_model.py fastflow/loss.py fastflow/anomaly_map.py \
  patchcore/torch_model.py patchcore/anomaly_map.py \
  padim/torch_model.py padim/anomaly_map.py \
  reverse_distillation/torch_model.py reverse_distillation/loss.py \
  reverse_distillation/anomaly_map.py reverse_distillation/components/bottleneck.py \
  components/backbone/resnet_decoder.py \
  components/flow/all_in_one_block.py components/base/dynamic_buffer.py \
  components/sampling/k_center_greedy.py \
  components/dimensionality_reduction/random_projection.py \
  components/filters/blur.py components/utils/deprecation.py \
  components/stats/multi_variate_gaussian.py
```

각 모델 디렉터리의 `__init__.py`는 이 프로젝트가 소유한 팩토리이므로 원본 대조 대상이 아니다. `reverse_distillation/LICENSE`는 원본 그대로의 라이선스 고지라 대조 대상이 아니다.

## 5. 신규 런타임 의존성

| 패키지 | 상태 (`pytorch_env`) | 사용처 | `requirements.txt` |
|---|---|---|---|---|
| `FrEIA` | 설치됨 | FastFlow `SequenceINN`, `all_in_one_block` | **미기재** |
| `kornia` | 0.8.2 | PatchCore·PaDiM `filters/blur.py` (`GaussianBlur2d`) | **미기재** |
| `scikit-learn` | 1.7.2 | PatchCore `random_projection.py` | **미기재** |
| `scipy` | 1.15.3 | `all_in_one_block.py` | **미기재** |
| `timm` | 1.0.22 | 공통 backbone | **미기재** |
| `omegaconf` | 2.3.0 | FastFlow `anomaly_map.py` 타입 힌트 | **미기재** |
| `tqdm` | 4.67.1 | `k_center_greedy.py` | 기재됨(간접) |

**PaDiM과 Reverse Distillation은 신규 의존성을 추가하지 않는다.** `multi_variate_gaussian.py`는 torch만 쓰고, RD의 `resnet_decoder.py`·`bottleneck.py`는 `torchvision.models.resnet`의 기본 블록만 재사용한다. `anomaly_map.py`가 쓰는 `omegaconf`는 FastFlow가 이미 끌어온 것이다.

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
| config | `configs/anomaly/models/patchcore.yaml` (`smooth_sigma: 0` — §8.4) |
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

## 8. 모델 연결 — PaDiM

| 항목 | 값 |
|---|---|
| upstream 파일 | `padim/torch_model.py`, `padim/anomaly_map.py` |
| 공유 components | `stats/multi_variate_gaussian.py` (신규), `base/dynamic_buffer.py`, `filters/blur.py`, `feature_extractors/timm.py`, `feature_extractors/utils.py`, `data/torch_base.py` |
| adapter | `adapters/padim.py#PadimAdapter` (`ADAPTERS: "padim"`) |
| 모델 팩토리 | `models/padim/__init__.py#build_padim` (`MODELS: "padim"`, `"padim_anomaly"`) |
| config | `configs/anomaly/models/padim.yaml` (`smooth_sigma: 0` — §8.4) |
| override한 hook | `train_step`, `on_validation_start` |
| 로컬 자산 | `${paths.backbone_root}/resnet18-f37072fd.pth` (기본) 또는 `wide_resnet50_2-95faca4d.pth` |
| 자산 주입 지점 | 팩토리. `pre_trained=False`로 생성 후 `feature_extractor.feature_extractor`(내부 timm 모듈)에 로드 |

### 8.1 `lightning_model.py` 이관 결과

| anomalib | 이 프로젝트 |
|---|---|
| `configure_optimizers` → **None** (최적화 없음) | config `optim`에 no-op optimizer 선언 (§8.2) |
| `training_step` → `self.model(batch.image)` + `torch.tensor(0.0, requires_grad=True)` | `PadimAdapter.train_step` (동일한 dummy loss) |
| `MemoryBankMixin.on_validation_start` → `fit()` → `PadimModel.fit()` | `PadimAdapter.on_validation_start` (§8.3) |
| `validation_step` → `InferenceBatch` 반환 | 공통 `AnomalyAdapter.eval_step` (override 불필요) |
| `trainer_arguments` → `max_epochs=1` | config `train.epochs: 1` |
| `trainer_arguments` → `val_check_interval=1.0`, `num_sanity_val_steps=0` | 대응물 없음. 공통 engine은 매 epoch 끝에 1회만 validation한다 |
| `trainer_arguments` → `devices=1` | 대응물 없음. 공통 engine은 단일 device 전용이다 |
| `configure_post_processor` → 기본 `PostProcessor()` | 공통 `AnomalyAdapter` + `postprocess/`가 담당 |
| `n_features` 기본값 (resnet18=100, wide_resnet50_2=550) | 원본 `_N_FEATURES_DEFAULTS`가 그대로 처리. config는 `null`이 기본이며 `selectors`가 명시값을 제공 |
| `learning_type` | 대응물 없음 |

`gradient_clip_val`은 PaDiM의 `trainer_arguments`에 없지만, gradient 학습이 없는 모델에 공통 engine의 clipping이 개입하지 않도록 config에 `train.grad_clip: null`을 명시했다.

### 8.2 optimizer — PatchCore와 동일하게 no-op

PaDiM도 gradient 학습을 하지 않는다. 처리 방식은 §7.2와 동일하며, 실측 결과도 같다.

- 팩토리는 백본을 **freeze하지 않는다.** 전부 freeze하면 `build_optimizer`가 빈 파라미터 리스트를 받는다.
- 실측: 전체 파라미터 2,782,784개, `requires_grad=True` 2,782,784개, optimizer param group 1개·텐서 45개. `loss.backward()` + `optimizer.step()` 후 **`.grad`가 non-None인 파라미터 0개**.
- `PadimModel.forward`가 feature 추출을 `torch.no_grad()`로 감싸고, `TimmFeatureExtractor.forward`가 매 호출 내부 timm 모듈에 `eval()`을 강제한다. 실측으로 `model.train()` 상태에서 forward해도 `bn1.running_mean`이 **변하지 않음**을 확인했다. 따라서 BatchNorm 통계 오염이 없고 EfficientAD식 `train` 바인딩 고정도 불필요하다.

### 8.3 Gaussian fitting 시점

`PadimModel.fit()`은 memory bank에 모인 embedding으로 위치별 다변량 Gaussian(`mean`, `inv_covariance`)을 추정한다. **첫 validation 직전**에 한 번만 실행해야 하며, `engine.py:52`의 `adapter.on_validation_start`가 정확히 그 자리다. anomalib의 `MemoryBankMixin.on_validation_start`와 대응한다.

- `_gaussian_fitted` 플래그로 1회만 수행한다. `epochs: 1`이 기본이라 실제로는 한 번만 불린다.
- `fit()`은 성공 후 `memory_bank`를 비운다. epoch를 2 이상으로 올린 경우 이후 epoch가 embedding을 다시 쌓지만 쓰이지 않으므로, adapter가 skip 분기에서 `model.memory_bank = []`로 비워 메모리가 무한히 늘지 않게 한다. 결과값에는 영향이 없다.
- **`fork_rng`는 쓰지 않았다.** PaDiM의 fitting과 Mahalanobis 스코어링은 난수를 전혀 소비하지 않는다. PaDiM의 유일한 난수 사용은 `PadimModel.__init__`의 `random.sample`(feature index 서브샘플)이며, 이는 생성 시점 1회로 run seed에 의해 고정되고 `idx` 버퍼로 checkpoint에 저장된다. 실측으로 `idx` shape `(100,)`, dtype `int64`를 확인했다.
- `gaussian.mean`·`gaussian.inv_covariance`는 `MultiVariateGaussian`이 `DynamicBufferMixin`을 상속해 가변 크기 buffer로 저장한다. checkpoint 재로드는 evaluate 스모크로 실제 복원을 확인했다.

### 8.4 blur 책임 분리 — 모델 내부 blur를 단일 출처로 삼는다

anomaly map의 Gaussian blur는 **모델이 스스로 정의한 것을 그대로 쓰고, postprocess smoother는 그것이 없는 모델에서만 따로 건다.** 같은 map에 blur가 두 번 걸리지 않게 하는 것이 원칙이다.

내부 blur를 가진 모델은 PaDiM·PatchCore·Reverse Distillation 셋이며, 모두 `sigma=4`를 기본으로 쓴다.

| 모델 | 모델 내부 blur | `adapter.params.smooth_sigma` |
|---|---|---|
| PaDiM | `GaussianBlur2d(sigma=4)` (`padim/anomaly_map.py:67`) | `0` |
| PatchCore | `GaussianBlur2d(sigma=4)` (`patchcore/anomaly_map.py:59`) | `0` |
| Reverse Distillation | `GaussianBlur2d(sigma=4)` (`reverse_distillation/anomaly_map.py:117`) | `0` |
| STFPM | 없음 | `4.0` |
| EfficientAD | 없음 | `4.0` |
| FastFlow | 없음 | `4.0` |

이 분리로 세 모델 모두 anomalib과 동등해진다. anomalib은 이들에 postprocess blur를 걸지 않는다.

#### 영향 범위는 pixel 지표로 한정된다

`AnomalyAdapter.eval_step`(`adapters/base.py:46-57`)에서 `pred_score`는 모델이 낸 값을 그대로 쓰고 `_smooth`를 거치지 않는다. `_smooth`가 닿는 것은 `anomaly_map`뿐이다. 따라서 `smooth_sigma`는 **pixel AUROC · `pixel_threshold` · 시각화**에만 영향을 주며, image AUROC는 어떤 모델에서도 이 값에 영향받지 않는다.

bottle 1 epoch 실측(valid):

| 모델 | `smooth_sigma` | image AUROC | pixel AUROC |
|---|---|---|---|---|
| PaDiM | `4.0` (변경 전) | 0.9973958 | 0.9808675 |
| PaDiM | `0` (변경 후) | 0.9973958 | 0.9808345 |
| PatchCore | `4.0` (변경 전) | 1.0000000 | 0.9850589 |
| PatchCore | `0` (변경 후) | 1.0000000 | 0.9853441 |

image AUROC는 두 모델 모두 비트 단위로 동일하다. pixel AUROC는 PaDiM이 3.3e-5 감소, PatchCore가 2.9e-4 증가로 양쪽 다 무시 가능한 수준이며, 이 변경의 근거는 성능이 아니라 **blur 출처의 단일화**다.

`smooth_sigma`는 `smoother.py:7`에서 `sigma is None or sigma <= 0`이면 map을 그대로 반환하므로 `0`이 곧 해제다.
### 8.5 가중치 로드 판정

`_load_backbone_weights`는 FastFlow·PatchCore와 동일한 규칙이며, PaDiM 전용 문구만 다르다.

- `missing`이 하나라도 있으면 `LocalAssetError`.
- `unexpected`는 extractor가 갖고 있지 않은 최상위 서브모듈만 허용한다. `resnet18` + `layers=[layer1, layer2, layer3]`는 `out_indices=[1,2,3]`으로 해석되어 `layer4`와 `fc`가 제거되므로 그 키들만 허용된다.
- shape 불일치(`RuntimeError`)는 `LocalAssetError`로 감싼다.
- 조용한 랜덤 초기화 폴백은 없다(CON-004).

세 모델이 이 헬퍼를 각자 복제하고 있다. 공통화한다면 `core/`가 아니라 anomaly task 내부 헬퍼로 올리는 것이 맞으며, 기존 파일 2개를 함께 고쳐야 하므로 별도 리팩터링 과제로 남긴다.

### 8.6 `_deduce_dims` — upstream 미사용 코드

`padim/torch_model.py`의 모듈 수준 함수 `_deduce_dims`와 그것이 쓰는 `dryrun_find_featuremap_dims` import는 `PadimModel` 어디에서도 호출되지 않는다. `PadimModel.__init__`은 `sum(self.feature_extractor.out_dims)`를 직접 쓴다. upstream 그대로의 상태이며 CON-001에 따라 삭제하지 않았다. import 대상인 `components/feature_extractors/utils.py`는 이미 존재해 추가 복사가 없었다.

## 9. 모델 연결 — Reverse Distillation

| 항목 | 값 |
|---|---|
| upstream 파일 | `reverse_distillation/torch_model.py`, `loss.py`, `anomaly_map.py`, `components/bottleneck.py` |
| 공유 components | `backbone/resnet_decoder.py` (신규), `feature_extractors/timm.py`, `filters/blur.py`, `data/torch_base.py` |
| adapter | `adapters/reverse_distillation.py#ReverseDistillationAdapter` (`ADAPTERS: "reverse_distillation"`) |
| 모델 팩토리 | `models/reverse_distillation/__init__.py#build_reverse_distillation` (`MODELS: "reverse_distillation"`, `"reverse_distillation_anomaly"`) |
| config | `configs/anomaly/models/reverse_distillation.yaml` (`smooth_sigma: 0` — §8.4) |
| override한 hook | `train_step`만 |
| 로컬 자산 | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` (기본), `resnet18-f37072fd.pth`, `resnet50-0676ba61.pth` |
| 자산 주입 지점 | 팩토리. `pre_trained=False`로 생성 후 `encoder.feature_extractor`(내부 timm 모듈)에 로드 |

구조는 encoder(고정) → OCBE 병목 → ResNet 디코더이며, 인코더와 디코더의 다중 스케일 feature를 코사인 유사도로 맞춘다. 이 레포에 encoder-decoder 계열이 처음 들어온 사례다.

### 9.1 `lightning_model.py` 이관 결과

| anomalib | 이 프로젝트 |
|---|---|
| `configure_optimizers` → `Adam(decoder.parameters() + bottleneck.parameters(), lr=0.005, betas=(0.5, 0.99))` | config `optim` + 팩토리의 encoder freeze (§9.2) |
| `training_step` → `self.loss(*self.model(batch.image))` | `ReverseDistillationAdapter.train_step` |
| `validation_step` → `InferenceBatch` 반환 | 공통 `AnomalyAdapter.eval_step` (override 불필요) |
| `trainer_arguments` → `gradient_clip_val=0` | config `train.grad_clip: null` |
| `trainer_arguments` → `num_sanity_val_steps=0` | 대응물 없음. 공통 engine에 sanity check 단계가 없다 |
| `input_size` 필수 (`self.input_size is None`이면 즉시 오류) | `model.params.input_size` (§9.3) |
| `anomaly_map_mode` 기본 `AnomalyMapGenerationMode.ADD` | `model.params.anomaly_map_mode: add` (§9.4) |
| `pre_trained=True` | 팩토리에서 `False` + 로컬 가중치 |
| `learning_type` | 대응물 없음 |

lifecycle hook이 하나도 필요 없다. FastFlow와 같은 부류로, adapter는 `train_step`만 override한다. memory bank도 학습 후 보정도 없다.

`max_epochs`는 `trainer_arguments`에 없어 anomalib이 값을 고정하지 않는다. config에는 RD4AD 논문의 200 epoch을 적고 잠정값임을 주석에 남겼다(§11).

### 9.2 optimizer — freeze로 파라미터 집합을 일치시킨다

anomalib은 `decoder`와 `bottleneck` 파라미터만 명시적으로 optimizer에 넣는다. 이 프로젝트의 `build_optimizer`(`builders.py:26`)는 `requires_grad=True`인 파라미터로 **단일 group**을 만들므로, 팩토리가 `build_optimizer` 실행 **이전에** encoder를 freeze하면 두 집합이 정확히 같아진다. `core/builders.py`는 건드리지 않았다.

실측으로 집합 동일성을 확인했다.

| 항목 | 값 |
|---|---|
| encoder 파라미터 | 24,862,528개 (`requires_grad=True`인 것 **0개**) |
| bottleneck | 39,222,272개 |
| decoder | 24,917,504개 |
| optimizer가 잡은 파라미터 | 64,139,776개 = bottleneck + decoder |
| `set(optimizer params) == set(decoder+bottleneck params)` | **True** |
| 1 step 후 grad가 붙은 텐서 | encoder 0개 / bottleneck+decoder 165개 |

`betas`는 YAML 리스트 `[0.5, 0.99]`로 전달되며 `build_adam`이 `**params`를 그대로 넘겨 `torch.optim.Adam`이 수용한다(실측 `param_groups[0]["betas"] == [0.5, 0.99]`).

`torch_model.forward`가 매 호출 `self.encoder.eval()`을 실행하므로(`torch_model.py:149`) engine의 per-epoch `model.train()`이 encoder를 학습 모드로 되돌리지 못한다. `model.train()` 상태에서 forward해도 `encoder.feature_extractor.bn1.running_mean`이 불변임을 실측했다. EfficientAD식 인스턴스 수준 `train` 바인딩 고정은 불필요하다.

### 9.3 `input_size` 중복 — FastFlow와 동일한 제약

`AnomalyMapGenerator`가 `image_size`로 anomaly map 캔버스를 미리 할당하므로 생성 시점에 크기가 필요하다. `build_model`이 `data.image_size`를 볼 수 없다는 §6.2의 제약이 그대로 적용되어 `model.params.input_size`에 상수로 적고 짝을 주석으로 명시했다.

FastFlow와 달리 **불일치가 즉시 예외로 드러나지 않을 수 있다.** FastFlow는 LayerNorm shape 오류로 바로 죽지만, RD는 `F.interpolate`로 거리 맵을 `image_size`에 맞추므로 잘못된 캔버스 위에서 조용히 점수가 계산될 여지가 있다. 두 값을 반드시 함께 바꾼다.

### 9.4 `anomaly_map_mode` 문자열 변환

`AnomalyMapGenerationMode`는 `str` Enum이고 config는 평문 `"add"`를 준다. 팩토리에서 `AnomalyMapGenerationMode(anomaly_map_mode)`로 정규화한다.

원시 문자열을 그대로 넘겨도 동작은 한다 — `str` 믹스인이 동등성과 해시를 값 기준으로 맞춰 주므로 upstream의 `mode not in {ADD, MULTIPLY}` 집합 조회를 통과한다(실측 확인). 변환의 이유는 **오타 시 유효값을 알려주며 즉시 실패**시키기 위함이지, 동작 보정이 아니다.

### 9.5 blur — 내부 blur 보유 모델

`anomaly_map.py:117`이 `GaussianBlur2d(sigma=4)`를 forward 안에서 생성해 적용한다. 따라서 §8.4의 원칙에 따라 `smooth_sigma: 0`이다. 내부 blur를 가진 세 번째 모델이다(PaDiM·PatchCore·Reverse Distillation).

upstream은 blur 모듈을 forward마다 새로 만들고 `.to(device)`한다. 비효율이지만 upstream 동작이므로 수정하지 않는다(CON-001).

### 9.6 가중치 로드 판정

`_load_backbone_weights`는 다른 세 모델과 동일한 규칙이다. `wide_resnet50_2` + `layers=[layer1, layer2, layer3]`는 `out_indices=[1,2,3]`으로 해석되어 `layer4`와 `fc`가 제거되므로 그 키들만 unexpected로 허용된다. `missing`이 있거나 shape이 어긋나면 `LocalAssetError`이며, 조용한 랜덤 초기화 폴백은 없다(CON-004).

## 10. 검증 기록

| 항목 | FastFlow | PatchCore | PaDiM | Reverse Distillation |
|---|---|---|---|---|
| 원본 diff | import 2줄 외 0라인 | import 6줄 외 0라인 | import 7줄 외 0라인 | import 6줄 외 0라인 |
| `grep -rn lightning` | 0건 | docstring `:class:` 참조 3건만 | docstring `:class:` 참조 2건만 (실행 경로 없음) | 0건 |
| `git status -- src/core/ scripts/` | 무변경 | 무변경 | 무변경 | 무변경 |
| 기존 모델 파일 | 무변경 | 무변경 | 무변경 | 무변경 |
| freeze/grad | extractor trainable 0개, 전체 3,510,528개 | 의도적 미freeze. `.grad is None` | 의도적 미freeze. 전체 2,782,784개, step 후 `.grad` non-None 0개 | encoder freeze(trainable 0개). optimizer 집합 == anomalib decoder+bottleneck (64,139,776개), 실측 True |
| BatchNorm 통계 오염 | 해당 없음 | 미측정 | `model.train()` 상태에서 `running_mean` 불변 확인 | `model.train()` 상태에서 `running_mean` 불변 확인 |
| train 스모크 (bottle) | 1 epoch → image 0.995 / pixel 0.954 | 1 epoch → image 1.000 / pixel 0.985 | 1 epoch → image 0.997 / pixel 0.981 | 2 epoch → image 0.995(e1) / pixel 0.978(e2) |
| evaluate 스모크 (bottle, test) | image 0.976 / pixel 0.965 | image 1.000 / pixel 0.989 | image 0.912 / pixel 0.985 | image 1.000 / pixel 0.976 |
| predict 스모크 (20장) | 정상 | 정상 | 정상 (`broken_large` 20장, 시각화 포함) | 정상 (`broken_large` 20장, 시각화 포함) |
| checkpoint 재로드 | 정상 | 가변 buffer 복원 정상 | `MultiVariateGaussian` 가변 buffer + `idx` 복원 정상 | 정상 (가변 buffer 없음) |

`src/core/`와 `scripts/`를 건드리지 않았으므로 기존 모델 재스모크는 불필요하다. PaDiM·Reverse Distillation 추가로 변경된 공용 파일은 `models/__init__.py`와 `adapters/__init__.py`의 export 추가뿐이며, 두 파일 모두 import가 정상 해석됨을 train 스모크가 확인한다. 레지스트리는 import 시점에 전 모델을 등록하므로, 스모크가 통과했다는 것은 여섯 모델 전부의 등록이 성립했다는 뜻이다.

PatchCore·PaDiM의 스모크 수치는 §8.4의 `smooth_sigma: 0` 적용 후 값이다. PatchCore는 기존 커밋 시점(`smooth_sigma: 4.0`) 대비 pixel만 바뀌었고(valid 0.98506 → 0.98534, test 0.988 → 0.989) image AUROC는 동일하다. STFPM·EfficientAD·FastFlow는 내부 blur가 없어 `4.0`을 유지하므로 영향이 없다.

PaDiM의 evaluate image AUROC 0.912는 valid(0.997)와 차이가 크다. 같은 checkpoint·같은 코드에서 split만 다르므로 split 구성 차이로 보이나, 3개 카테고리 정식 검증에서 재확인이 필요하다(§11).

## 11. 미완 항목

- FastFlow `train.epochs: 100`은 잠정값이다. 핀된 클론에 `examples/configs`가 sparse-checkout되어 있지 않아 anomalib의 공식 학습 예산을 확인하지 못했다.
- 3개 카테고리(bottle, carpet, capsule) 성능 검증 — **사용자 실행 대기**. PaDiM·Reverse Distillation 포함.
- PaDiM evaluate(test) image AUROC 0.912의 원인 확인 — split 구성 차이 가설 검증 필요 (§10).
- 반대 벤더 CLI 적대적 검증 — 미실행 (FastFlow·PatchCore·PaDiM·Reverse Distillation 모두).
- `requirements.txt`에 `FrEIA`·`kornia`·`scikit-learn`·`scipy`·`timm`·`omegaconf` 누락 (§5).
- FastFlow의 cait/deit 백본 미지원. 로컬 자산이 HF safetensors 디렉터리라 현재 `torch.load` 경로로 읽히지 않는다.
- PatchCore·PaDiM은 `runtime.amp: true`와 호환되지 않을 수 있다 (§7.2). 검증하지 않았다.
- `_load_backbone_weights`가 FastFlow·PatchCore·PaDiM·Reverse Distillation 4곳에 복제되어 있다 (§8.5). 네 번째 복제로 공통화 필요성이 뚜렷해졌으나, 기존 파일 3개를 함께 고쳐야 하므로 별도 과제다.
- Reverse Distillation `train.epochs: 200`은 RD4AD 논문 기준 잠정값이다. anomalib이 `max_epochs`를 고정하지 않아 공식 예산을 확인하지 못했다 (§9.1). 스모크 2 epoch에서 image AUROC가 0.995(e1) → 0.740(e2)로 크게 흔들렸다 — 초기 학습 변동으로 보이나 정식 학습에서 수렴 확인이 필요하다.
- Reverse Distillation은 `input_size`와 `data.image_size` 불일치가 즉시 예외로 드러나지 않을 수 있다 (§9.3).
- `postprocess/__init__.py`가 8개 이름을 re-export하지만 이를 경유하는 import가 한 곳도 없다. 실사용 공개 API는 `smooth_anomaly_map`·`to_output_dict`·`compute_thresholds`·`save_prediction_visualization` 4개이며, `best_f1_threshold`는 `compute_thresholds` 내부 헬퍼다. 공통 코드라 정리는 별도 과제다.

---

작성일: 2026-08-23
문서 상태: FastFlow·PatchCore·PaDiM·Reverse Distillation 추가 산출물 (anomalib `091ca6a` 기준)
