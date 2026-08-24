# UPSTREAM-INVENTORY (v0.2) — 복사 원본 출처·라이선스 인벤토리

선행 문서: [v0.1 UPSTREAM-INVENTORY](../../v0.1/reports/UPSTREAM-INVENTORY.md) · 절차: [MODEL-ADD](../../v0.1/reports/MODEL-ADD.md)

v0.2에서 추가된 anomalib 복사본을 기록한다. 대상은 **FastFlow**, **PatchCore**, **PaDiM**, **Reverse Distillation**, **DFM**, **DFKDE**, **CFA**이며, STFPM·EfficientAD 행은 v0.1 문서가 SSOT이라 여기서 반복하지 않는다.

**갱신 이력**: FastFlow·PatchCore·PaDiM·Reverse Distillation은 최초 작성 시점 기준. DFM·DFKDE·CFA는 §4.5~4.7, §4.8~4.9(갱신), §5(갱신), §10~12, §13~14(갱신)에서 이어서 기록한다.

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

### 4.5 DFM

| 대상 | 원본 | 줄 수 | sha256(16) | 허용 변경 |
|---|---|---|---|---|
| `dfm/torch_model.py` | `src/anomalib/models/image/dfm/torch_model.py` | 240 | `6df0c11e8d80f837` | import 4건 |
| `components/dimensionality_reduction/pca.py` | `src/anomalib/models/components/dimensionality_reduction/pca.py` | 197 | `5098aabc22c09c6e` | import 1건 |

재사용한 기존 components는 `feature_extractors/timm.py`, `base/dynamic_buffer.py`, `data/torch_base.py`이며 신규 복사는 `pca.py` 하나뿐이다.

### 4.6 DFKDE

| 대상 | 원본 | 줄 수 | sha256(16) | 허용 변경 |
|---|---|---|---|---|
| `dfkde/torch_model.py` | `src/anomalib/models/image/dfkde/torch_model.py` | 172 | `5ea1ba1242cde831` | import 3건 |
| `components/classification/kde_classifier.py` | `src/anomalib/models/components/classification/kde_classifier.py` | 244 | `c01165f9b3b06ed6` | import 2건 |
| `components/stats/kde.py` | `src/anomalib/models/components/stats/kde.py` | 147 | `c21bfa18c5f44ece` | import 1건 |

DFM의 `pca.py`를 재사용한다(신규 복사 아님). `kde_classifier.py`가 `pca.py`와 `kde.py` 양쪽을 가져온다.

### 4.7 CFA

| 대상 | 원본 | 줄 수 | sha256(16) | 허용 변경 |
|---|---|---|---|---|
| `cfa/torch_model.py` | `src/anomalib/models/image/cfa/torch_model.py` | 572 | `d475d192566722b0` | import 3건 |
| `cfa/anomaly_map.py` | `src/anomalib/models/image/cfa/anomaly_map.py` | 131 | `49628da66e4af4a9` | import 1건 |
| `cfa/loss.py` | `src/anomalib/models/image/cfa/loss.py` | 87 | `ac2d8711bc540152` | 없음 |

CFA는 `TimmFeatureExtractor`를 쓰지 않는다. `torch_model.py`의 module-level `get_feature_extractor`가 `torchvision.models.feature_extraction.create_feature_extractor`로 `torchvision.models.<backbone>`을 직접 감싼다 — PatchCore·PaDiM·Reverse Distillation과 다른 feature extraction 경로이며, no-download 처리도 다르다(§12.2). `components/`에서 신규로 재사용한 파일은 없다(`base/dynamic_buffer.py`, `filters/blur.py`, `data/torch_base.py`, `feature_extractors/utils.py` 모두 기존 파일 재사용).

세 모델 모두 `__init__.py`와 `lightning_model.py`는 복사하지 않았다(§4.3과 동일한 사유). `dfm`·`dfkde`·`cfa` 세 디렉터리 모두 `__init__.py`를 두지 않는 기존 관례를 유지한다(팩토리 파일이 `models/<name>/__init__.py` 자체이므로 이 규칙과는 별개다).

### 4.8 FRE

| 대상 | 원본 | 줄 수 | sha256(16) | 허용 변경 |
|---|---|---|---|---|
| `fre/torch_model.py` | `src/anomalib/models/image/fre/torch_model.py` | 199 | `85e4aec9495baf85` | import 2건 |

FRE는 기존 `TimmFeatureExtractor`와 `InferenceBatch`를 재사용하며 추가 components 복사는 없다.

### 4.9 U-Flow

| 대상 | 원본 | 줄 수 | sha256(16) | 허용 변경 |
|---|---|---|---|---|
| `uflow/torch_model.py` | `src/anomalib/models/image/uflow/torch_model.py` | 297 | `dfeef07735ad9891` | import 2건 |
| `uflow/feature_extraction.py` | `src/anomalib/models/image/uflow/feature_extraction.py` | 291 | `f1ff1c3691cb947e` | import 1건 |
| `uflow/anomaly_map.py` | `src/anomalib/models/image/uflow/anomaly_map.py` | 219 | `f96eb65332e3dfb8` | 없음 |
| `uflow/loss.py` | `src/anomalib/models/image/uflow/loss.py` | 72 | `ccd6e11739906b59` | 없음 |

U-Flow는 기존 `AllInOneBlock`, `TimmFeatureExtractor`, `InferenceBatch`를 재사용하며, 자체 `feature_extraction.py`(`LayerNormFeatureExtractor`)를 포함한다.

### 4.10 CS-Flow

| 대상 | 원본 | 줄 수 | sha256(16) | 허용 변경 |
|---|---|---|---|---|
| `csflow/torch_model.py` | `src/anomalib/models/image/csflow/torch_model.py` | 744 | `69900db7e5e5f7b3` | import 2건 |
| `csflow/anomaly_map.py` | `src/anomalib/models/image/csflow/anomaly_map.py` | 104 | `abd36a06b43c636c` | 없음 |
| `csflow/loss.py` | `src/anomalib/models/image/csflow/loss.py` | 53 | `c02d47c082b4e794` | 없음 |

CS-Flow는 기존 `TimmFeatureExtractor`와 `InferenceBatch`를 재사용하며 추가 components 복사는 없다.

### 4.11 SuperSimpleNet

| 대상 | 원본 | 줄 수 | sha256(16) | 허용 변경 |
|---|---|---|---|---|
| `supersimplenet/torch_model.py` | `src/anomalib/models/image/supersimplenet/torch_model.py` | 399 | `83cf00867c0f5bf1` | import 3건 |
| `supersimplenet/anomaly_generator.py` | `src/anomalib/models/image/supersimplenet/anomaly_generator.py` | 162 | `71a8cfc7b539daa1` | import 1건 |
| `supersimplenet/loss.py` | `src/anomalib/models/image/supersimplenet/loss.py` | 75 | `fa3381a2cf4847fe` | 없음 |
| `components/generators/perlin.py` | `src/anomalib/data/utils/generators/perlin.py` (함수 추출) | 153 | `8c9ee270a517c85d` | 없음 |

SuperSimpleNet은 기존 `TimmFeatureExtractor`, `GaussianBlur2d`, `InferenceBatch`를 재사용하며, `generate_perlin_noise` 함수를 `components/generators/perlin.py`로 분리하여 포함한다.

### 4.12 허용된 변경 — import 경로 전량

CON-001이 허용하는 유일한 변경이다. 아래가 전부이며, 그 외 차이는 원본과 0라인이다.

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
| `dfm/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `dfm/torch_model.py` | `...components.dimensionality_reduction.pca import PCA` |
| `dfm/torch_model.py` | `...components.base.dynamic_buffer import DynamicBufferMixin` |
| `dfm/torch_model.py` | `...components.feature_extractors.timm import TimmFeatureExtractor` |
| `components/dimensionality_reduction/pca.py` | `...components.base.dynamic_buffer import DynamicBufferMixin` |
| `dfkde/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `dfkde/torch_model.py` | `...components.feature_extractors.timm import TimmFeatureExtractor` |
| `dfkde/torch_model.py` | `...components.classification.kde_classifier import FeatureScalingMethod, KDEClassifier` |
| `components/classification/kde_classifier.py` | `...components.dimensionality_reduction.pca import PCA` / `...components.stats.kde import GaussianKDE` (2줄, §4.9와 같은 사유로 분리) |
| `components/stats/kde.py` | `...components.base.dynamic_buffer import DynamicBufferMixin` |
| `cfa/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `cfa/torch_model.py` | `...components.base.dynamic_buffer import DynamicBufferMixin` |
| `cfa/torch_model.py` | `...components.feature_extractors.utils import dryrun_find_featuremap_dims` |
| `cfa/anomaly_map.py` | `...components.filters.blur import GaussianBlur2d` |
| `fre/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `fre/torch_model.py` | `...components.feature_extractors.timm import TimmFeatureExtractor` |
| `uflow/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `uflow/torch_model.py` | `...components.flow.all_in_one_block import AllInOneBlock` |
| `uflow/feature_extraction.py` | `...components.feature_extractors.timm import TimmFeatureExtractor` |
| `csflow/torch_model.py` | `...components.data.torch_base import InferenceBatch` |
| `csflow/torch_model.py` | `...components.feature_extractors.timm import TimmFeatureExtractor` |
| `supersimplenet/torch_model.py` | `...components.data.torch_base import InferenceBatch` / `...components.filters.blur import GaussianBlur2d` / `...components.feature_extractors.timm import TimmFeatureExtractor` / `from .anomaly_generator import AnomalyGenerator` |
| `supersimplenet/anomaly_generator.py` | `...components.generators.perlin import generate_perlin_noise` |

`patchcore/torch_model.py`의 원본 한 줄 `from anomalib.models.components import DynamicBufferMixin, KCenterGreedy, TimmFeatureExtractor`는 패키지 `__init__`을 경유하면 Lightning이 딸려 오므로 **모듈 단위 3줄로 분리**했다. `padim/torch_model.py`의 `from anomalib.models.components import MultiVariateGaussian, TimmFeatureExtractor`도 같은 이유로 2줄로 분리했다. `components/classification/kde_classifier.py`의 `from anomalib.models.components import PCA, GaussianKDE`도 동일한 사유로 2줄로 분리했다. 줄 수가 늘지만 성격은 import 경로 치환이다.

### 4.13 무결성 확인

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
  components/stats/multi_variate_gaussian.py \
  dfm/torch_model.py dfkde/torch_model.py \
  cfa/torch_model.py cfa/anomaly_map.py cfa/loss.py \
  components/dimensionality_reduction/pca.py \
  components/classification/kde_classifier.py components/stats/kde.py \
  fre/torch_model.py \
  uflow/torch_model.py uflow/feature_extraction.py uflow/anomaly_map.py uflow/loss.py \
  csflow/torch_model.py csflow/anomaly_map.py csflow/loss.py \
  supersimplenet/torch_model.py supersimplenet/anomaly_generator.py supersimplenet/loss.py \
  components/generators/perlin.py
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
| `tqdm` | 4.67.1 | `k_center_greedy.py`, CFA `initialize_centroid` | 기재됨(간접) |
| `einops` | 0.8.1 | CFA `torch_model.py`·`anomaly_map.py` (`rearrange`) | **미기재** |

**PaDiM과 Reverse Distillation은 신규 의존성을 추가하지 않는다.** `multi_variate_gaussian.py`는 torch만 쓰고, RD의 `resnet_decoder.py`·`bottleneck.py`는 `torchvision.models.resnet`의 기본 블록만 재사용한다. `anomaly_map.py`가 쓰는 `omegaconf`는 FastFlow가 이미 끌어온 것이다. **DFM·DFKDE도 신규 의존성을 추가하지 않는다** — `pca.py`·`kde.py`·`kde_classifier.py` 전부 torch만 쓴다. **CFA는 `einops` 하나가 새로 추가되고, `scikit-learn`(`sklearn.cluster.KMeans`, `gamma_c > 1`일 때만 호출)과 `torchvision`은 이미 다른 경로로 확보되어 있다.**

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

## 10. 모델 연결 — DFM

| 항목 | 값 |
|---|---|
| upstream 파일 | `dfm/torch_model.py` |
| 공유 components | `dimensionality_reduction/pca.py` (신규), `base/dynamic_buffer.py`, `feature_extractors/timm.py`, `data/torch_base.py` |
| adapter | `adapters/dfm.py#DfmAdapter` (`ADAPTERS: "dfm"`) |
| 모델 팩토리 | `models/dfm/__init__.py#build_dfm` (`MODELS: "dfm"`, `"dfm_anomaly"`) |
| config | `configs/anomaly/models/dfm.yaml` (`smooth_sigma: 4.0` — §10.3) |
| override한 hook | `train_step`, `on_validation_start` |
| 로컬 자산 | `${paths.backbone_root}/resnet50-0676ba61.pth` (기본), selector로 `resnet18`·`wide_resnet50_2` |

### 10.1 `lightning_model.py` 이관 결과

| anomalib | 이 프로젝트 |
|---|---|
| `configure_optimizers` → **None** | config `optim`에 no-op optimizer 선언 (PatchCore·PaDiM과 동일, §7.2 참조) |
| `training_step` → feature 수집 + dummy loss | `DfmAdapter.train_step` |
| `MemoryBankMixin.on_validation_start` → `fit()` → PCA(+`nll`이면 Gaussian) | `DfmAdapter.on_validation_start` |
| `validation_step` → `InferenceBatch` 반환 | 공통 `AnomalyAdapter.eval_step` (override 불필요, `fre` 모드 한정) |
| `trainer_arguments` → `max_epochs=1`, `gradient_clip_val=0` | config `train.epochs: 1`, `train.grad_clip: null` |

### 10.2 optimizer — PatchCore·PaDiM과 동일하게 no-op

DFM도 gradient 학습이 없다. `_load_backbone_weights` 판정, freeze 미적용(빈 파라미터 리스트 회피), `.grad is None` 실측까지 PatchCore(§7.2)와 완전히 동일한 근거를 공유한다. 별도 조사는 하지 않았다.

### 10.3 `score_type` — anomaly map 존재 여부가 갈린다

- `score_type="fre"`(기본): PCA 재구성 오차를 pixel-level anomaly map으로 만든다. 내부 blur가 없으므로(PatchCore·PaDiM·Reverse Distillation과 다름) `smooth_sigma`는 공통 기본값 `4.0`을 유지한다.
- `score_type="nll"`: Gaussian NLL이며 anomaly map을 만들지 않는다. 이 프로젝트의 공통 pixel-metric pipeline과 연결돼 있지 않으므로, 이 모드로 evaluate를 실행하면 `AnomalyAdapter.eval_step`이 `None` anomaly map을 `smooth_anomaly_map`에 넘겨 예외가 난다. 사용하려면 DFKDE 방식(§11)의 adapter override가 추가로 필요하다 — 미구현.

### 10.4 upstream 버그 — `SingleClassGaussian.fit`의 `torch.mean(..., device=...)`

`torch_model.py:73`의 `torch.mean(dataset, dim=1, device=dataset.device)`는 `torch.mean`이 받지 않는 `device` 키워드를 넘겨 `TypeError`를 낸다. `score_type="nll"`일 때만 호출되는 경로이며 기본값 `"fre"`에서는 도달하지 않는다. upstream 원본 그대로이므로 수정하지 않는다(CON-001). `nll`을 쓰려면 이 버그가 먼저 upstream에서 해결되어야 한다.

## 11. 모델 연결 — DFKDE

| 항목 | 값 |
|---|---|
| upstream 파일 | `dfkde/torch_model.py` |
| 공유 components | `classification/kde_classifier.py` (신규), `stats/kde.py` (신규), `dimensionality_reduction/pca.py`(DFM과 공유), `feature_extractors/timm.py`, `data/torch_base.py` |
| adapter | `adapters/dfkde.py#DfkdeAdapter` (`ADAPTERS: "dfkde"`) |
| 모델 팩토리 | `models/dfkde/__init__.py#build_dfkde` (`MODELS: "dfkde"`, `"dfkde_anomaly"`) |
| config | `configs/anomaly/models/dfkde.yaml` (`metrics: [image_auroc]`만 — §11.2) |
| override한 hook | `train_step`, `on_validation_start`, `eval_step`, `update_metrics`, `predict_step`, `on_fit_end` |
| 로컬 자산 | `${paths.backbone_root}/resnet18-f37072fd.pth` (기본), selector로 `wide_resnet50_2` |

### 11.1 `lightning_model.py` 이관 결과

DFM(§10.1)과 optimizer·fit 시점 구조는 동일하다. 차이는 `configure_evaluator`가 `image_auroc`·`image_f1score`만 등록한다는 점 — DFKDE는 이 프로젝트 여섯 모델 중 유일하게 anomaly map을 만들지 않는 모델이며, §11.2에서 그 결과를 다룬다.

### 11.2 anomaly map 부재 — 공통 adapter를 그대로 쓸 수 없다

`DfkdeModel.forward`는 추론 시 `InferenceBatch(pred_score=scores)`만 반환한다(`anomaly_map` 인자 없음, 기본값 `None`). 공통 `AnomalyAdapter.eval_step`/`predict_step`/`on_fit_end`(threshold 계산)는 모두 `outputs["anomaly_map"]`이 존재한다고 가정하고 `smooth_anomaly_map`·`.flatten()`을 호출하므로 `None`에서 그대로 크래시한다.

`DfkdeAdapter`가 세 hook을 전부 재정의해 pixel 경로를 건너뛴다.

- `eval_step`: `maps`/`masks` 자체를 계산하지 않고 `scores`/`labels`만 담아 반환한다.
- `update_metrics`: `image_auroc`만 갱신한다.
- `on_fit_end`: `postprocess.smoother.compute_thresholds`를 호출하지 않고, 그 안의 image-score 절반만 `best_f1_threshold`로 직접 재구현해 `image_threshold`를 계산한다. `pixel_threshold`는 `None`으로 고정한다.
- `predict_step`: base 구현에서 `maps = self._smooth(...)`와 `self._last_maps = maps.detach().cpu()` 두 줄만 제거한 버전이다. `self._last_maps`가 `__init__`의 초기값 `None`으로 남으므로 상속받은 `visualize()`가 `if ... self._last_maps is None: return`으로 안전하게 no-op된다 — 별도 override 불필요.

config의 `metrics:` 목록도 `image_auroc` 하나로 전체 교체해 `pixel_auroc`를 제거한다. `deep_merge`가 리스트를 병합이 아니라 교체하므로 이 방식이 성립한다(data config는 손대지 않는다 — EfficientAD의 `data:` 블록 오버라이드와 같은 메커니즘, `docs/guides/anomaly-models.md` §3.8).

### 11.3 `KDEClassifier.fit`의 subsample — `torch` RNG와 무관

`kde_classifier.py:159`의 `random.sample(...)`은 Python 표준 `random` 모듈을 쓴다(`max_training_points=40000` 초과 시에만 호출). `torch.random.fork_rng`가 감싸는 대상이 아니므로 PatchCore·PaDiM식 guard를 두지 않았다. `random.seed()`는 `runtime.seed`로 별도 시드된다(`src/core/context.py`).

## 12. 모델 연결 — CFA

| 항목 | 값 |
|---|---|
| upstream 파일 | `cfa/torch_model.py`, `cfa/anomaly_map.py`, `cfa/loss.py` |
| 공유 components | `base/dynamic_buffer.py`, `filters/blur.py`, `feature_extractors/utils.py`, `data/torch_base.py` (전부 기존 재사용, 신규 없음) |
| adapter | `adapters/cfa.py#CfaAdapter` (`ADAPTERS: "cfa"`) |
| 모델 팩토리 | `models/cfa/__init__.py#build_cfa` (`MODELS: "cfa"`, `"cfa_anomaly"`) |
| config | `configs/anomaly/models/cfa.yaml` (`smooth_sigma: 0`, `train.epochs: 30` 잠정 — §15) |
| override한 hook | `on_fit_start`, `train_step` |
| 로컬 자산 | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` (기본), selector로 `resnet18` |

### 12.1 `lightning_model.py` 이관 결과

| anomalib | 이 프로젝트 |
|---|---|
| `on_train_start` → `initialize_centroid(train_dataloader)` | `CfaAdapter.on_fit_start` (§12.3) |
| `training_step` → `distance = model(x)`; `loss = self.loss(distance)` | `CfaAdapter.train_step` |
| `backward` override (`retain_graph=True`) | 필요 없음 — 원인이 다르며 adapter가 다르게 해결한다 (§12.4) |
| `configure_optimizers` → `AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4, amsgrad=True)` | config `optim` (§12.2) |
| `validation_step` → `InferenceBatch` 반환 | 공통 `AnomalyAdapter.eval_step` (override 불필요) |
| `trainer_arguments` → `gradient_clip_val=0` | config `train.grad_clip: null` |

### 12.2 feature extraction — `TimmFeatureExtractor`가 아니다

`torch_model.py`의 module-level `get_feature_extractor(backbone, return_nodes)`가 `getattr(torchvision.models, backbone)(pretrained=True)` 뒤 `torchvision.models.feature_extraction.create_feature_extractor`로 감싼다. PatchCore·PaDiM·Reverse Distillation·DFM·DFKDE는 전부 `TimmFeatureExtractor`를 쓰므로 CFA만 별도 no-download 처리가 필요했다.

- 팩토리가 `torchvision.models.<backbone>` 모듈 속성을 `pretrained=False` wrapper로 **일시 치환**하고 `try/finally`로 원복한다(STFPM의 `timm.create_model` 치환과 같은 기법, 대상만 다르다).
- `get_feature_extractor`는 매 호출 `getattr(torchvision.models, backbone)`을 새로 하므로 속성 치환만으로 충분하고 `torch_model.py`는 건드리지 않는다.
- 가중치는 `create_feature_extractor`가 반환한 `GraphModule`에 직접 로드한다. FX 추출은 서브모듈 이름을 바꾸지 않으므로 state-dict 키가 원본 backbone과 같다 — PatchCore·PaDiM과 같은 `_load_backbone_weights` 판정 규칙(missing 즉시 실패, unexpected는 없는 서브모듈만 허용)을 그대로 적용했다.
- 팩토리가 backbone을 `requires_grad=False`로 명시적으로 고정한다. anomalib은 고정하지 않지만 `forward`가 항상 `torch.no_grad()`로 backbone을 실행해 그레이디언트가 애초에 생기지 않으므로 결과는 동일하다 — PatchCore·PaDiM과 달리 CFA는 descriptor network(CoordConv2d, trainable 4개 tensor)가 있어 freeze해도 `build_optimizer`가 빈 리스트를 받지 않는다.
- **적대적 검토에서 지적된 한계**: 이 치환은 전역 `torchvision.models` 속성을 lock 없이 바꾸므로 여러 스레드에서 동시에 `build_cfa`를 호출하면 이론적으로 다른 스레드의 wrapper가 뒤섞이거나 원복이 어긋날 수 있다. 이 프로젝트의 CLI는 모델을 항상 직렬로 생성하므로 실제 발생 경로는 없다. STFPM의 `timm.create_model` 치환(v0.1 문서)도 동일한 성격의 위험을 안고 있어, 새로운 문제가 아니라 기존에 받아들인 위험의 반복이다.

### 12.3 centroid 초기화 — `.image` 속성 브리지

`initialize_centroid(data_loader)`는 `for i, data in enumerate(tqdm(data_loader)): batch = data.image.to(device)`로 anomalib의 `Batch`(`.image` 속성 보유)를 직접 기대한다. 이 프로젝트의 loader는 `(images, targets)` 튜플을 낸다.

모델 파일을 고치는 대신 `adapters/cfa.py`에 `_ImageBatch`(슬롯 하나짜리 `.image` wrapper)와 `_as_image_batches(loader)`(제너레이터)를 두어 `CfaAdapter.on_fit_start`가 `model.initialize_centroid(data_loader=_as_image_batches(loaders["train"]))`로 호출한다. `initialize_centroid`가 읽는 필드는 `.image` 하나뿐이므로 이 wrapper로 원본 계산(전체 train set 순회 평균 → `gamma_c > 1`이면 k-means)이 그대로 재현된다. 합성 배치로 실측 확인함(`memory_bank` shape이 backbone 채널 수와 일치).

### 12.4 학습 loop 버그 — `CfaLoss.radius`가 그래프를 붙들고 있었다 (사용자 재현, 수정함)

첫 학습 실행에서 두 번째 배치의 `backward()`가 다음으로 실패했다.

```text
RuntimeError: Trying to backward through the graph a second time
```

원인: `loss.py:58`의 `self.radius = torch.ones(1, requires_grad=True) * radius`는 leaf가 아니라 `grad_fn`을 가진 non-leaf 텐서다. `CfaAdapter.__init__`이 애초에 `CfaLoss`를 **한 번만** 만들었으므로 모든 학습 스텝이 이 텐서 하나를 공유했고, 첫 `backward()`가 그 계산 그래프를 해제해 두 번째 스텝이 죽었다.

anomalib은 `Cfa.backward`를 `loss.backward(retain_graph=True)`로 오버라이드해 우회한다(`lightning_model.py:233`, "Investigate why retain_graph is needed"라는 TODO가 붙어 있다 — 바로 이 원인이다). 이 프로젝트의 공통 engine(`engine.py:130`)은 모델별 backward 훅이 없고, 매 스텝 전체 그래프를 유지하는 것은 메모리 누수로 이어진다.

**조치**: `CfaAdapter.train_step`이 매 스텝 `self.cfa_loss.radius = torch.ones(1, requires_grad=True) * self._radius`로 텐서를 새로 만든다. `radius`는 `nn.Parameter`가 아니고 `build_optimizer`에도 전달되지 않으므로(CFA는 backbone을 freeze해 descriptor만 optimizer에 들어간다, §12.2) 이전 스텝에 쌓였을 그레이디언트가 적용된 적이 없다 — 수치적으로 anomalib과 동일하며 매 스텝 새 leaf 노드를 갖는다는 차이만 있다. SSOT 파일(`loss.py`)은 수정하지 않았다.

4스텝 연속 backward로 재현·수정을 확인했다(§13).

### 12.5 model/loss 하이퍼파라미터 이중 선언 (적대적 검토 지적, 수정함)

최초 구현은 `num_nearest_neighbors`·`num_hard_negative_features`·`radius`를 `model.params`와 `adapter.params` 양쪽에 각각 선언했다. anomalib의 `Cfa.__init__`은 하나의 인자 집합을 `CfaModel`과 `CfaLoss` 양쪽에 전달하므로 불일치가 원천적으로 불가능한데, 이 구조는 두 config 값이 어긋나면 anomaly map과 학습 손실이 다른 하이퍼파라미터를 쓰는 조용한 버그가 될 수 있었다.

**조치**: `CfaAdapter.on_fit_start`가 이미 생성된 `model`에서 `model.num_nearest_neighbors`·`model.num_hard_negative_features`·`model.radius`를 직접 읽어 `CfaLoss`를 구성하도록 바꿨다. `cfa.yaml`의 `adapter.params`에서 세 값을 제거했다 — 이제 `model.params`가 유일한 출처다.

## 13. 검증 기록

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

PaDiM의 evaluate image AUROC 0.912는 valid(0.997)와 차이가 크다. 같은 checkpoint·같은 코드에서 split만 다르므로 split 구성 차이로 보이나, 3개 카테고리 정식 검증에서 재확인이 필요하다(§15).

### 13.1 DFM·DFKDE·CFA

| 항목 | DFM | DFKDE | CFA |
|---|---|---|---|
| 원본 diff | import 4줄 외 0라인 | import 3줄 외 0라인 | import 3줄 외 0라인 |
| `grep -rn lightning` | 0건 | 0건 | 0건 (docstring의 "Lightning" 설명 문구 제외) |
| `git status -- src/core/` | 무변경 | 무변경 | 무변경 |
| 기존 모델 파일(STFPM~RD) | 무변경 | 무변경 | 무변경 |
| registry 등록 | `MODELS.build("dfm_anomaly")` 성공 | `MODELS.build("dfkde_anomaly")` 성공 | `MODELS.build("cfa_anomaly")` 성공 |
| freeze/grad | 의도적 미freeze(PatchCore·PaDiM과 동일 사유) | 의도적 미freeze(동일) | backbone freeze, trainable 4개 tensor(descriptor)만 optimizer에 포함 — 실측 |
| 오프라인 재현 | `--print_config` 정상, local weight 강제 로드 확인 | 동일 | `torch.hub` 캐시 숨긴 상태에서 local weight만으로 빌드 성공(다운로드 없음 확인) |
| `--print_config` | 정상 (`metrics:` 기본 유지) | 정상 (`metrics: [image_auroc]`로 교체됨 확인) | 정상 (`adapter.params`에 하이퍼파라미터 없음 확인, §12.5) |
| 합성 배치 스모크 | — | — | `on_fit_start`→centroid 초기화→4스텝 연속 `train_step`+`backward()` 성공, eval `pred_score`/`anomaly_map` shape 확인 |
| 반대 벤더 적대적 검토 | Codex CLI 1회, Critical 0 | Codex CLI 1회, Critical 0 | Codex CLI 1회, Critical 0, Major 2건 수정(§12.4, §12.5) — 상세는 `docs/dev/v0.2/reviews/A1.md` |
| train/evaluate/predict 실행 | 사용자 실행 확인(2026-08-23) | 사용자 실행 확인(2026-08-23) | 사용자 실행 확인(2026-08-23, §12.4 수정 후) |
| 수치 기록(image/pixel AUROC 등) | 미수집 | 미수집 | 미수집 |

DFM·DFKDE·CFA 세 모델 모두 사용자가 `scripts/train.py`·`evaluate.py`·`predict.py`를 직접 실행해 정상 동작을 확인했다(2026-08-23). 실제 MVTec 데이터로 학습을 가장 먼저 돌린 것은 CFA이며, 그 첫 실행에서 §12.4의 `retain_graph` 버그가 실제로 재현되었다 — 합성 스모크가 놓친 결함이었다. 세 모델 모두 image/pixel AUROC 등 구체적인 수치는 아직 기록되지 않았고, 3개 카테고리(bottle·carpet·capsule) 기준 정식 성능 비교는 별도 진행 예정이다(§15 참조).

## 14. 모델 연결 — CFLOW

### 14.1 `lightning_model.py` 이관 결과

Upstream `Cflow`는 `automatic_optimization=False`로 선언하고 `training_step`에서 이미지 배치 하나를 fiber(`fiber_batch_size` 단위 feature-vector 조각)로 잘라 fiber마다 독립적으로 `opt.zero_grad()`/`self.manual_backward(loss.mean())`/`opt.step()`을 호출한다(기본 config·MVTec 256×256·batch 8 기준 배치당 약 168회 step). `configure_optimizers`는 decoder 파라미터만 모은 단일 Adam(lr=0.0001)을 반환한다. `validation_step`은 `self.model(batch.image)`만 호출해 공통 `AnomalyAdapter.eval_step`으로 흡수된다. `trainer_arguments`는 `gradient_clip_val=0`, `num_sanity_val_steps=0`이며 별도 calibration lifecycle은 없다.

이 프로젝트의 공통 engine은 `adapter.train_step()` 1회당 정확히 1회의 zero_grad+backward+step만 수행하므로, fiber당 다회 step 구조를 그대로 옮길 수 없었다. `CflowAdapter`(`src/tasks/anomaly/adapters/cflow.py`)가 decoder 파라미터 전용 private `torch.optim.Adam`을 `on_fit_start`에서 만들고 `train_step` 내부에서 upstream과 동일한 순서로 fiber마다 zero_grad/backward/step을 수행한다. engine에 반환하는 loss는 `0.0 * parameter`를 decoder 전체에 합산한 더미 텐서라, engine 소유 optimizer(같은 decoder 파라미터를 대상으로 `build_optimizer`가 만든 것)는 항상 정확히 0인 gradient만 받아 Adam 모멘트가 영원히 0으로 유지되고 step이 no-op이 된다. 상세 근거와 검증은 `docs/dev/v0.2/reviews/A2.md` 참조.

### 14.2 optimizer — decoder 전용 private optimizer (STFPM~CFA와 다른 패턴)

기존 9개 모델은 모두 `build_optimizer`(공통 engine 소유, `model.parameters()` 중 `requires_grad=True`)만으로 학습이 성립했다. CFLOW는 fiber당 다회 step이 필요해 이 패턴이 성립하지 않아, adapter가 자체 optimizer를 갖는 첫 사례가 됐다. encoder는 기존 관례대로 factory(`build_cflow`)에서 `requires_grad=False`로 고정되며, `TimmFeatureExtractor.forward()`가 `requires_grad=False`일 때 내부적으로 `eval()`+`no_grad()`를 강제하므로 EfficientAD 방식의 instance-level `train()` 고정은 불필요하다(실측 확인).

### 14.3 checkpoint resume 미지원 (알려진 한계)

`CflowAdapter`의 private optimizer는 `state_dict()`/`load_state_dict()`를 구현하지 않고, 공통 engine도 `adapter_state`를 채워 넣지 않는다(`src/core/checkpoint.py`의 `adapter_state` 매개변수는 존재하나 `src/core/engine.py`가 사용하지 않는 기존 미완성 확장점). `--resume`으로 CFLOW를 이어서 학습하면 decoder 가중치·RNG는 정상 복원되지만 Adam 모멘텀은 0부터 다시 시작한다. 공통 engine 변경이 필요해 이번 세션 범위 밖으로 이월했다(§15).

### 14.4 가중치 로드 판정

`_load_backbone_weights`가 FastFlow·PatchCore·PaDiM·Reverse Distillation·DFM·DFKDE에 이어 CFLOW에도 복제됐다(§8.5, §15 참조 — 공통화는 별도 과제). 로드 대상은 `model.encoder.feature_extractor`(TimmFeatureExtractor 내부의 raw timm 모듈)이며, `model.encoder`(wrapper) 자체를 넘기면 `feature_extractor.` 접두사 불일치로 missing keys 오류가 난다는 것을 실제로 재현·수정하며 확인했다(PaDiM과 동일 패턴, `build_padim`의 주석 참조).

## 15. 모델 연결 — FRE

### 15.1 `lightning_model.py` 이관 결과

Upstream `Fre`는 `configure_optimizers`에서 `optim.Adam(params=self.model.fre_model.parameters(), lr=1e-3)`를 반환하고, `training_step`에서 CNN backbone 추출 feature와 Tied AutoEncoder(`TiedAE`) 재구성 feature 간 `MSELoss`를 계산한다. `validation_step`은 `self.model(batch.image)`를 호출하여 anomaly map과 score를 반환하며, `trainer_arguments`는 `gradient_clip_val=0`, `num_sanity_val_steps=0`이다.

이 프로젝트에서는:
- Optimizer: config `optim.optimizer` (Adam, `lr: 0.001`)
- Training step: `FreAdapter.train_step`에서 `model.get_features(images)`를 호출하고 원본 feature와 재구성 feature 간 `torch.nn.MSELoss()` 계산
- Validation / Prediction: 공통 `AnomalyAdapter.eval_step` 및 `predict_step`으로 흡수
- Backbone: `build_fre` 팩토리에서 `pre_trained=False`로 구성 후 로컬 가중치 strict 로드 및 `requires_grad=False`로 backbone 고정. `TiedAE` 파라미터만 학습 대상으로 등록

### 15.2 스모크 검증 결과

- `train.py` (MVTec bottle, 1 epoch): 정상 완료, loss 0.007, valid pixel_auroc 0.965
- `evaluate.py` (MVTec bottle, test): test image_auroc 0.500, pixel_auroc 0.973
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료

## 16. 모델 연결 — U-Flow

### 16.1 `lightning_model.py` 이관 결과

Upstream `Uflow`는 `configure_optimizers`에서 `optim.Adam(params=self.parameters(), lr=1e-3, weight_decay=1e-5)` 및 `LinearLR(start_factor=1.0, end_factor=0.4, total_iters=25000)`를 반환하고, `training_step`에서 multi-scale feature를 U자형 flow로 변환한 latent variables `z`와 log-Jacobian determinant `ljd`를 받아 `UFlowLoss()(z, ljd)`를 계산한다. `validation_step`은 `self.model(batch.image)`를 호출하여 anomaly map과 score를 반환하며, `trainer_arguments`는 `num_sanity_val_steps=0`이다.

이 프로젝트에서는:
- Optimizer: config `optim.optimizer` (Adam, `lr: 0.001`, `weight_decay: 0.00001`)
- Training step: `UflowAdapter.train_step`에서 `model(images)`를 호출하고 `UFlowLoss()(z, ljd)` 계산
- Validation / Prediction: 공통 `AnomalyAdapter.eval_step` 및 `predict_step`으로 흡수
- Backbone: `build_uflow` 팩토리에서 `timm.create_model`을 `pretrained=False`로 래핑하여 초기화 후, 로컬 가중치 strict 로드 및 `requires_grad=False`로 backbone 고정. `feature_normalizations`(`LayerNorm`) 및 `flow`(`GraphINN`) 파라미터만 학습 대상으로 등록

### 16.2 스모크 검증 결과

- `train.py` (MVTec bottle, 1 epoch): 정상 완료, loss 947730.039, valid image_auroc 0.734, pixel_auroc 0.864
- `evaluate.py` (MVTec bottle, test): test image_auroc 0.673, pixel_auroc 0.899
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료 (1.4 img/s)

## 17. 모델 연결 — CS-Flow

### 17.1 `lightning_model.py` 이관 결과

Upstream `Csflow`는 `configure_optimizers`에서 `optim.Adam(self.parameters(), lr=2e-4, eps=1e-4, weight_decay=1e-5, betas=(0.5, 0.9))`를 반환하고, `training_step`에서 multi-scale feature를 cross-scale flow로 변환한 `z_dist`와 log-Jacobian determinant `jacobians`를 받아 `CsFlowLoss()(z_dist, jacobians)`를 계산한다. `validation_step`은 `self.model(batch.image)`를 호출하여 anomaly map과 score를 반환하며, `trainer_arguments`는 `gradient_clip_val=1`, `num_sanity_val_steps=0`이다.

이 프로젝트에서는:
- Optimizer: config `optim.optimizer` (Adam, `lr: 0.0002`, `eps: 0.0001`, `weight_decay: 0.00001`, `betas: [0.5, 0.9]`)
- Training step: `CsflowAdapter.train_step`에서 `model(images)`를 호출하고 `CsFlowLoss()(z_dist, jacobians)` 계산
- Validation / Prediction: 공통 `AnomalyAdapter.eval_step` 및 `predict_step`으로 흡수
- Backbone: `build_csflow` 팩토리에서 `torchvision.models.efficientnet_b5`를 `weights=None`으로 래핑하여 초기화 후, 로컬 가중치 strict 로드 및 `requires_grad=False`로 backbone 고정. `CrossScaleFlow` 파라미터만 학습 대상으로 등록

### 17.2 스모크 검증 결과

- `train.py` (MVTec bottle, 1 epoch): 정상 완료, loss 12.026, valid image_auroc 0.951, pixel_auroc 0.500
- `evaluate.py` (MVTec bottle, test): test image_auroc 0.931, pixel_auroc 0.500
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료 (1.4 img/s)

## 18. 모델 연결 — SuperSimpleNet

### 18.1 `lightning_model.py` 이관 결과

Upstream `Supersimplenet`는 `configure_optimizers`에서 `AdamW([{"params": self.model.adaptor.parameters(), "lr": 1e-4}, {"params": self.model.segdec.parameters(), "lr": 2e-4, "weight_decay": 1e-5}])` 및 `MultiStepLR(milestones=[0.8*epochs, 0.9*epochs], gamma=0.4)`를 반환하고, `training_step`에서 `model(images, masks, labels)`를 호출하여 feature-level 합성 이상을 생성한 `anomaly_map`, `anomaly_score`, `target_masks`, `target_labels`를 받아 `SSNLoss()(pred_map, pred_score, target_mask, target_label)`를 계산한다. `validation_step`은 `self.model(batch.image)`를 호출하여 anomaly map과 score를 반환하며, `trainer_arguments`는 `gradient_clip_val=0`, `num_sanity_val_steps=0`이다.

이 프로젝트에서는:
- Optimizer: config `optim.optimizer` (AdamW, `lr: 0.0002`, `weight_decay: 0.00001`)
- Training step: `SupersimplenetAdapter.train_step`에서 `model(images, masks, labels)`를 호출하고 `SSNLoss` 계산
- Validation / Prediction: 공통 `AnomalyAdapter.eval_step` 및 `predict_step`으로 흡수
- Backbone: `build_supersimplenet` 팩토리에서 `timm.create_model`을 `pretrained=False`로 래핑하여 초기화 후, 로컬 가중치 strict 로드 및 `requires_grad=False`로 backbone 고정. `adaptor`(`projection`) 및 `segdec`(`seg_head`, `cls_conv`, `cls_fc`) 파라미터만 학습 대상으로 등록

### 18.2 스모크 검증 결과

- `train.py` (MVTec bottle, 1 epoch): 정상 완료, loss 1.555, valid image_auroc 0.724, pixel_auroc 0.578
- `evaluate.py` (MVTec bottle, test): test image_auroc 0.562, pixel_auroc 0.615
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료 (1.2 img/s)

## 20. 모델 연결 — GANomaly

### 20.1 `lightning_model.py` 이관 결과

Upstream `Ganomaly`는 `configure_optimizers`에서 Generator용 `Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))` 및 Discriminator용 `Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))` 두 개의 옵티마이저를 반환하며, `automatic_optimization=False`로 Generator와 Discriminator를 각각 수동 스텝한다. `training_step`에서 Generator 손실(Adversarial + Contextual + Latent Error)과 Discriminator 손실(BCE)을 계산한다. `validation_step`은 `self.model(batch.image)`를 호출하여 image-level anomaly score `pred_score`를 반환하며 (pixel anomaly map 없음), `trainer_arguments`는 `num_sanity_val_steps=0`이다.

이 프로젝트에서는:
- Optimizer: `GanomalyAdapter`가 private `optimizer_g`, `optimizer_d`를 생성 및 스케줄링하고 Engine에는 dummy 0 loss를 반환
- Training step: `GanomalyAdapter.train_step`에서 generator와 discriminator 순차 forward-backward-step 수행
- Validation / Prediction: 공통 `AnomalyAdapter.eval_step` 및 `predict_step`으로 흡수 (`anomaly_map is None` 지원)
- Backbone: 백본 없음 (from scratch 학습)

### 20.2 스모크 검증 결과

- `train.py` (MVTec bottle, 1 epoch): 정상 완료, loss 0.169, valid image_auroc 0.177
- `evaluate.py` (MVTec bottle, test): test image_auroc 0.177
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 결과 저장 정상 완료 (1.5 img/s)

## 21. 모델 연결 — DRAEM

### 21.1 `lightning_model.py` 이관 결과

Upstream `Draem`은 `configure_optimizers`에서 `optim.Adam(self.model.parameters(), lr=1e-4)` 및 `MultiStepLR(milestones=[400, 600], gamma=0.1)`를 반환하고, `training_step`에서 DTD/Perlin 노이즈 기반 `PerlinAnomalyGenerator`로 합성 결함 이미지를 생성한 뒤 `DraemLoss()(input_image, reconstruction, anomaly_mask, prediction)`(L2 + SSIM + Focal)를 계산한다. `validation_step`은 `self.model(batch.image)`를 호출하여 anomaly map과 score를 반환하며, `trainer_arguments`는 `gradient_clip_val=0`, `num_sanity_val_steps=0`이다.

이 프로젝트에서는:
- Optimizer: config `optim.optimizer` (Adam, `lr: 0.0001`) 및 `optim.scheduler` (multistep, `milestones: [400, 600]`, `gamma: 0.1`)
- Training step: `DraemAdapter.train_step`에서 `PerlinAnomalyGenerator`로 증강 후 `DraemLoss` 계산
- Validation / Prediction: 공통 `AnomalyAdapter.eval_step` 및 `predict_step`으로 흡수
- Texture Dataset: 로컬 `${paths.dataset_root}/dtd` 경로에서 오프라인 로드

### 21.2 스모크 검증 결과

- `train.py` (MVTec bottle, 1 epoch): 정상 완료, loss 1.172, valid image_auroc 0.375, pixel_auroc 0.745
- `evaluate.py` (MVTec bottle, test): test image_auroc 0.419, pixel_auroc 0.730
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료 (1.1 img/s)

## 22. 모델 연결 — DSR

### 22.1 `lightning_model.py` 이관 결과

Upstream `Dsr`은 `configure_optimizers`에서 Phase 1(Reconstruction + Anomaly Detection)용 `optimizer_d`(`Adam(lr=2e-4)`) + `StepLR`, Phase 2(Upsampling)용 `optimizer_u`(`Adam(lr=2e-4)`)를 반환하고, `automatic_optimization=False`로 단계별로 학습 모듈을 전환한다. Phase 1에서는 `DsrAnomalyGenerator`로 quantized feature defect를 생성하여 `DsrSecondStageLoss`를 계산하고, Phase 2에서는 `PerlinAnomalyGenerator`로 smudge 이상을 생성하여 `DsrThirdStageLoss`를 계산한다. `validation_step`은 `self.model(batch.image)`를 호출하여 anomaly map과 score를 반환하며, `trainer_arguments`는 `num_sanity_val_steps=0`이다.

이 프로젝트에서는:
- Optimizer: `DsrAdapter`가 private `optimizer_d`, `scheduler_d`, `optimizer_u`를 관리하고 `second_phase_epoch` 도달 시 Phase 1 모듈 freeze 및 Phase 2 모듈 unfreeze
- Training step: `DsrAdapter.train_step`에서 현재 epoch에 따라 Phase 1 / Phase 2 분기 학습 수행
- Validation / Prediction: 공통 `AnomalyAdapter.eval_step` 및 `predict_step`으로 흡수
- Pretrained Codebook: 로컬 `${paths.backbone_root}/vq_model_pretrained_128_4096.pckl`에서 오프라인 로드

### 22.2 스모크 검증 결과

- `train.py` (MVTec bottle, 1 epoch): 정상 완료, loss 0.001, valid image_auroc 0.427, pixel_auroc 0.609
- `evaluate.py` (MVTec bottle, test): test image_auroc 0.622, pixel_auroc 0.625
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료 (1.1 img/s)

## 23. 사용자 실제 학습에서 드러난 결함 3건 (2026-08-24, 모두 수정함)

사용자가 GANomaly·DSR·CS-Flow를 capsule/bottle에 10 epoch씩 실제로 돌려 "성능이 개선되지 않는다"고 보고했다. 세 건 모두 원인이 달랐고, 그중 둘은 실제 결함이었다.

### 23.1 GANomaly — 결함 아님 (모델 자체의 한계)

측정값: bottle best image AUROC 0.490, capsule best 0.851. 대부분의 epoch에서 0.5 미만.

anomalib 공식 벤치마크(`ganomaly/README.md`)의 MVTec 평균 image AUROC가 **0.421**이고 15개 카테고리 중 12개가 0.5 미만이다(bottle **0.251**, capsule **0.682**). 우리 수치는 오히려 reference를 상회한다. `loss.py`는 upstream과 바이트 단위 동일하고 `train_step`도 upstream `training_step`과 라인 단위로 일치하며 하이퍼파라미터도 전부 기본값과 같다. upstream의 min-max 정규화(`_normalize`)는 단조 변환이라 AUROC를 바꾸지 않으므로 미포팅이어도 무해하다.

원인은 구조적이다. GANomaly는 이미지 전체를 100차원 latent로 압축하고 그 재구성 오차 `‖z − ẑ‖²`만으로 점수를 내는데, MVTec 결함은 국소적이라 전역 병목에 흔적을 거의 남기지 않는다. 남는 신호는 밝기·자세 같은 nuisance 변동이 지배하고 그것이 라벨과 역상관을 이루면 0.5 아래로 떨어진다. 원래 MNIST/CIFAR의 "클래스 하나를 통째로 이상치로 간주"하는 설정용 모델이다.

**조치 없음.** MVTec 실사용 후보로는 부적합하며, 0.5 미만이 정상임을 여기에 기록해 같은 의문이 반복되지 않게 한다.

### 23.2 DSR — Phase 1이 통째로 스킵되고 있었다 (Critical, 수정함)

증상: image AUROC가 10 epoch 내내 소수점 3자리까지 완전 고정(capsule 0.587, bottle 0.427)인데 pixel AUROC만 변동. epoch 소요 시간도 전부 ~39s로 균일.

`adapters/dsr.py`의 `on_fit_start`가 `total_epochs = getattr(self, "total_epochs", 1)`로 값을 읽는데, **`total_epochs`는 레포 전체에서 이 `getattr` 한 곳에만 존재하고 대입되는 곳이 없었다.** 항상 fallback `1` → `second_phase_epoch = int(1 × 0.7) = 0`. 그 결과 `train_step`의 `current_epoch <= second_phase_epoch`가 항상 거짓이 되어 epoch 1부터 phase 2만 실행됐고, `on_epoch_start`가 epoch 1에 즉시 `image_reconstruction_network`·`subspace_restriction_module_hi`·`_lo`·`anomaly_detection_module`을 모두 동결했다. DSR 핵심 모듈 4개가 랜덤 초기화 상태로 박제된 채 `upsampling_module`만 학습된 것이다. `optimizer_d`는 생성만 되고 한 번도 쓰이지 않았다.

증상이 정확히 맞아떨어지는 이유는 `torch_model.py`의 eval 경로에 있다 — image 점수는 `anomaly_detection_module` 출력(`out_mask_sm`)에서, pixel 맵은 `upsampling_module` 출력(`out_mask_sm_up`)에서 나온다. 전자는 동결, 후자는 학습 중이었다.

upstream(`lightning_model.py:166-170`)은 `num_steps = max(max_steps // len(loader), trainer.max_epochs)`로 계산해 10 epoch면 `second_phase = 7`이다. 인덱싱 규약 차이(upstream 0-index `<` vs 이 프로젝트 1-index `<=`)는 등가라 문제가 없었고, 값만 공급되면 해결되는 사안이었다.

**조치**: `src/core/engine.py`의 `fit()`이 `adapter.on_fit_start()` 직전에 `adapter.total_epochs = ctx.epochs`를 설정한다. 모델명 분기가 아닌 일반 속성이므로 NFR-005 위반이 아니며, 이 값을 쓰지 않는 adapter는 영향받지 않는다.

검증(capsule 10 epoch, `metrics_epoch.csv`):

| | 수정 전 | 수정 후 |
|---|---|---|
| image AUROC | 10 epoch 내내 0.587 고정 | epoch 1~7 변동 (0.447→0.667→0.577→0.417→0.619→0.573→0.550) |
| epoch 소요 | 전부 ~39s | phase1 ~155s / phase2 ~34s |
| best image AUROC | 0.587 | 0.667 |

독립적 확증 두 가지: epoch 8부터 image AUROC가 `0.5502645373344421`로 고정되는데(phase 2는 `upsampling_module`만 학습하므로 정상) 이는 phase 경계 7/8과 정확히 일치하고, 같은 구간에서 pixel AUROC는 0.551→0.757→0.802로 상승한다. 또 epoch 시간이 155s에서 34s로 급감하는데, 수정 전 로그가 전 epoch ~39s였다는 사실이 곧 처음부터 phase 2만 돌았다는 증거다.

**이 결함이 V-05 스모크를 통과한 이유**: `--epochs 1`이면 `int(1 × 0.7) = 0`이라 **정상 코드에서도** phase 1이 실행되지 않는다(upstream도 동일한 축퇴 동작). 1-epoch 스모크로는 원리적으로 검출 불가능했다. **다단계 학습 모델은 스모크 epoch 수를 phase 경계보다 크게 잡아야 한다.**

### 23.3 DSR — upstream이 금지하는 ImageNet 정규화가 적용되고 있었다 (Major, 수정함)

upstream `Dsr.on_train_start`는 `Normalize`가 있으면 `ValueError: Transforms for DSR should not contain Normalize.`로 즉시 실패한다. pretrained VQ-VAE 코드북(`vq_model_pretrained_128_4096.pckl`)이 [0,1] 이미지로 학습됐기 때문이다. 그런데 `dsr.yaml`에는 `data:` 블록이 없었고 `anomaly_default`는 `normalize=True`가 기본이라 ImageNet 정규화가 그대로 들어갔다. EfficientAD는 같은 제약을 `efficientad.yaml`에서 `normalize: false`로 처리하고 있는데 DSR만 누락됐다.

**조치**: `configs/anomaly/models/dsr.yaml`에 `data.transform.{train,eval}.params.normalize: false`를 추가하고 근거를 주석으로 남겼다.

### 23.4 AUROC 지표의 sigmoid 포화 (Critical, 수정함) — 모델 무관, 지표 계층 결함

증상: CS-Flow의 pixel AUROC가 모든 epoch에서 **정확히 0.500**. 반면 image AUROC는 0.593→0.728→0.812로 정상 학습 중이었고 loss도 15.2→5.8로 건강하게 감소했다.

`metrics/{image,pixel}_auroc.py`가 `torchmetrics.classification.BinaryAUROC`를 그대로 쓰는데, **torchmetrics는 `preds`가 [0,1]을 벗어나면 자동으로 sigmoid를 적용한다.** AUROC는 순위 기반이고 sigmoid는 단조증가라 보통은 무해하지만, float32에서 sigmoid는 대략 `x > 17`부터 정확히 `1.0`으로 포화한다. 포화하는 순간 모든 값이 동점이 되어 결과가 정확히 0.5로 붕괴한다.

학습된 CS-Flow checkpoint 실측(capsule valid): anomaly map 범위 **511 ~ 21,700**, 고유값 404,820개, inf/nan 0개, 마스크 양성 3,591픽셀 — 맵 자체는 완전히 정상인데 `sigmoid(511) = sigmoid(21700) = 1.0`이라 전 픽셀이 한 점으로 뭉갰다. CS-Flow만 이렇게 큰 이유는 anomaly map이 3개 스케일 `mean(z²)`의 **곱**이기 때문이다. image AUROC가 멀쩡했던 것은 `pred_score`가 4.53~4.88이라 sigmoid 후에도 구분됐기 때문이다.

재현 실험(동일한 순위 정보, 값 범위만 변경): 511~21,700 → AUROC **0.5000**, 같은 순위를 [0,1]로 스케일 → **1.0000**.

**조치**: `src/tasks/anomaly/metrics/rank_auroc.py`에 `RankAUROC`를 추가하고 `image_auroc`·`pixel_auroc` 빌더를 여기로 교체했다. Mann-Whitney U 통계량을 동점 보정(tied group 평균 순위)과 함께 계산하며 입력을 어떤 변환도 없이 그대로 받는다. 누적 상태는 CPU에 두어 full-resolution pixel 지표가 GPU 메모리를 잠식하지 않는다.

**중요 — 기존 기록 수치 일부가 부풀려져 있었다.** 처음에는 "sigmoid가 단조증가이므로 포화만 피하면 기존 수치는 유효하다"고 판단했으나, 실측 결과 **틀렸다**. PaDiM은 `pred_score` 범위가 8.11~49.42라 일부가 포화하고 있었다:

| 모델 (bottle, 1 epoch) | 점수 범위 | 구 BinaryAUROC | 신 RankAUROC |
|---|---|---|---|
| PaDiM image | 8.11 ~ 49.42 | 0.997396 | **0.994792** |
| PaDiM pixel | 0.09 ~ 49.42 | 0.980834 | **0.981105** |
| PatchCore image/pixel | 1.07 ~ 4.28 | 1.000000 / 0.985344 | 동일 |
| CFLOW image/pixel | 0.06 ~ 2.50 | 1.000000 / 0.984500 | 동일 |
| CS-Flow pixel (capsule, test) | 511 ~ 21,700 | 0.500 | **0.646** |

포화는 "정상 이미지가 이상 이미지보다 높게 나온 오답"을 동점(0.5점)으로 바꾸므로 일반적으로 AUROC를 **부풀린다**. 즉 신 지표 값이 정확하고 구 값이 낙관적이었다.

합성 검증에서 `RankAUROC`는 비포화 구간(멀티배치 누적 포함) 전부에서 `BinaryAUROC`와 소수점 6자리까지 일치했고, 단일 클래스(→0.0)·전체 동점(→0.5) 엣지 케이스도 동일하다. 따라서 이 교체는 **정확하던 수치는 그대로 두고 틀렸던 수치만 고친다.**

**영향 범위**: 점수 또는 맵이 대략 17을 넘는 모델. PaDiM·CS-Flow는 확인됐고, 나머지 모델은 §23.5에 재측정 과제로 남긴다.

### 23.5 회귀 검증

core(`engine.py`)와 공용 지표를 모두 건드렸으므로 대표 모델 회귀 스모크를 돌렸다(bottle, 1 epoch).

| 모델 | 기록값 | 재측정 | 판정 |
|---|---|---|---|
| PatchCore | image 1.000 / pixel 0.985 | image 1.000 / pixel 0.985 | 일치 |
| CFLOW | image 1.000 / pixel 0.985 | image 1.000 / pixel 0.985 | 일치 |
| PaDiM | image 0.997 / pixel 0.981 | image 0.995 / pixel 0.981 | image는 §23.4의 지표 교정분 |

## 24. 모델 연결 — UniNet

### 24.1 `lightning_model.py` 이관 결과

Upstream `UniNet`은 `configure_optimizers`에서 `AdamW([{"params": self.model.student.parameters()}, {"params": self.model.bottleneck.parameters()}, {"params": self.model.dfs.parameters()}, {"params": self.model.teachers.target_teacher.parameters(), "lr": 1e-6}], lr=5e-3, betas=(0.9,0.999), weight_decay=1e-5, eps=1e-10, amsgrad=True)`와 `MultiStepLR(milestones=[0.8*max_steps_or_epochs], gamma=0.2)`를 반환한다. `training_step`은 `self.model(images=batch.image, masks=batch.gt_mask, labels=batch.gt_label)`을 호출해 loss 텐서를 직접 받고, `validation_step`은 `self.model(batch.image)`를 호출해 `InferenceBatch`를 받는다. `trainer_arguments`는 빈 dict다.

이 프로젝트에서는:
- Optimizer: `UniNetAdapter.configure_optimizers`가 upstream과 동일한 4-group split-LR AdamW를 그대로 구성 (`multistep` 스케줄러는 `core/builders.py`에 이미 존재하는 범용 빌더 재사용)
- Training step: `UniNetAdapter.train_step`. MVTec train split이 빈 target(`{}`)이라 mask/label이 없는 경우 all-zero-label로 대체 — all-normal 배치에서 실제 all-zero per-pixel mask와 수학적으로 동치임을 확인함(§A5)
- Validation / Prediction: 공통 `AnomalyAdapter.eval_step`/`predict_step`으로 흡수
- Backbone: `build_uninet`이 `torchvision.models.<backbone>`을 생성 구간에서만 `pretrained=False`로 치환해 오프라인 생성 후, source/target teacher 양쪽에 동일 로컬 가중치를 strict 로드. `source_teacher`는 `requires_grad=False` 고정 + 인스턴스 수준 `train` 오버라이드로 engine의 매 epoch `model.train()` 재귀에도 eval 고정 유지 (EfficientAD teacher와 동일 패턴)

### 24.2 스모크 및 10-epoch 검증 결과

- `train.py` (MVTec bottle, 1 epoch): 정상 완료, loss 16.046, valid image_auroc 0.979, pixel_auroc 0.891
- `evaluate.py` (MVTec bottle, test): image_auroc 0.979, pixel_auroc 0.914
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료 (1.2 img/s)
- 10-epoch 학습 (bottle): image_auroc 2 epoch부터 1.000 포화, pixel_auroc 10 epoch 시점 0.989

## 25. 모델 연결 — Dinomaly

### 25.1 `lightning_model.py` 이관 결과

Upstream `Dinomaly`는 `configure_optimizers`에서 `total_steps`(= `max_steps` 또는 `max_epochs × len(train_dataloader)`)를 동적으로 계산해 `StableAdamW([{"params": self.trainable_modules.parameters()}], lr=2e-3, betas=(0.9,0.999), weight_decay=1e-4, amsgrad=True, eps=1e-8)`와 `WarmCosineScheduler(base_value=2e-3, final_value=2e-4, total_iters=total_steps, warmup_iters=100)`를 매 optimizer step(배치)마다 `.step()`하는 구조로 반환한다. `training_step`은 `self.model(batch.image, global_step=self.global_step)`을 호출한다. `__init__`에서 전체 파라미터를 freeze한 뒤 `bottleneck`/`decoder`만 unfreeze하고 truncated-normal(Linear)/constant(LayerNorm) 초기화를 적용한다. `configure_pre_processor`는 `Resize(448)→CenterCrop(392)→Normalize(ImageNet)`을 사용한다.

이 프로젝트에서는:
- Optimizer/Scheduler: 공통 engine이 `scheduler.step()`을 epoch당 1회만 호출해 upstream의 per-step 스케줄과 맞지 않으므로, `DinomalyAdapter`가 `StableAdamW`+`WarmCosineScheduler`를 private으로 소유하고 매 배치 zero_grad→backward→step→scheduler.step→**zero_grad**를 직접 수행한 뒤 zero dummy loss를 engine에 반환한다(`CflowAdapter`와 동일 패턴). 적대적 검토(A5)에서 private step 후 `.grad`를 재차 비우지 않아 engine 소유 optimizer가 stale gradient로 이중 업데이트하는 Major 결함이 발견되어 즉시 수정했다.
- Freeze/초기화: `build_dinomaly` 팩토리가 upstream `Dinomaly.__init__`과 동일한 순서(전체 freeze → bottleneck/decoder unfreeze → trunc_normal/LayerNorm 초기화)로 재현
- Training step: `DinomalyAdapter.train_step`
- Backbone: `DinoV2Loader`가 캐시 디렉터리를 `torch.hub.get_dir()/dinov2`로 하드코딩하므로, `build_dinomaly`가 생성 구간에서만 `DinoV2Loader.__init__`을 몽키패치해 로컬 `paths.backbone_root`로 전환. 적대적 검토(A5)에서 sibling 파일이 없을 때 조용히 네트워크 다운로드로 폴백할 수 있는 Critical 결함이 발견되어, `DinoV2Loader`의 이름 파싱/경로 결정 로직을 재사용하는 `components/dinov2/local_preflight.py`로 생성 전 강제 검증을 추가했다.
- 전처리: `data.image_size=[392,392]` 단일 Resize로 upstream의 Resize(448)+CenterCrop(392)를 근사(현재 `anomaly_default` transform에 crop 단계 없음) — 알려진 편차로 config에 명시

### 25.2 스모크 및 10-epoch 검증 결과

- `train.py` (MVTec bottle, 1 epoch, batch_size 4): 정상 완료, loss 0.378, valid image_auroc 1.000, pixel_auroc 0.965
- `evaluate.py` (MVTec bottle, test): image_auroc 0.987, pixel_auroc 0.966
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료 (1.1 img/s)
- 10-epoch 학습 (bottle): image_auroc 2 epoch부터 1.000 포화, pixel_auroc 10 epoch 시점 0.990 (gradient-누수 수정 후 재검증)

## 26. 모델 연결 — AnomalyDINO

### 26.1 `lightning_model.py` 이관 결과

Upstream `AnomalyDino`는 gradient 학습이 없는 memory-bank 모델이다. `training_step`은 `self.model(batch.image)`로 정규화된 patch feature를 embedding store에 수집만 하고, `on_train_epoch_end`에서 `self.model.fit()`을 호출해 memory bank를 확정한다(옵션에 따라 coreset subsampling 포함). `validation_step`은 `self.model(batch.image)`로 kNN 코사인 거리 기반 `InferenceBatch`를 반환한다.

이 프로젝트에서는:
- Training step: `AnomalyDinoAdapter.train_step`이 PatchCore와 동일하게 dummy 0 loss(`requires_grad=True`인 detached leaf)를 반환
- Memory bank 확정: `AnomalyDinoAdapter.on_validation_start`가 `model.fit()`을 1회 호출 (`model.memory_bank.numel() > 0`이면 skip) — PatchCore의 `subsample_embedding` 시점과 동일
- Backbone: `AnomalyDINOModel`이 `DinoV2Loader.from_name(encoder_name)`을 호출하는 구조라 Dinomaly와 동일한 몽키패치 + `local_preflight` 방식을 재사용
- Anomaly map: PatchCore의 `AnomalyMapGenerator`와 `KCenterGreedy`를 그대로 import해 재사용 (신규 컴포넌트 복제 없음)

### 26.2 스모크 검증 결과

- `train.py` (MVTec bottle, 1 epoch, batch_size 4, fit-only): 정상 완료, image_auroc 1.000, pixel_auroc 0.985
- `evaluate.py` (MVTec bottle, test): image_auroc 1.000, pixel_auroc 0.991
- `predict.py` (MVTec bottle test broken_large 20장): 추론 및 시각화 저장 정상 완료 (1.2 img/s)
- fit-only 구조라 10-epoch 재학습은 수행하지 않음 (PatchCore와 동일하게 1 epoch로 완결)

## 27. 미완 항목

- FastFlow `train.epochs: 100`은 잠정값이다. 핀된 클론에 `examples/configs`가 sparse-checkout되어 있지 않아 anomalib의 공식 학습 예산을 확인하지 못했다.
- 3개 카테고리(bottle, carpet, capsule) 기준 정식 성능 비교(수치 기록) — **사용자 실행 대기**. PaDiM·Reverse Distillation·DFM·DFKDE·CFA·CFLOW·FRE·U-Flow·CS-Flow·SuperSimpleNet·GANomaly·DRAEM·DSR 포함.
- PaDiM evaluate(test) image AUROC 0.912의 원인 확인 — split 구성 차이 가설 검증 필요 (§13).
- `requirements.txt`에 `FrEIA`·`kornia`·`scikit-learn`·`scipy`·`timm`·`omegaconf`·`einops` 누락 (§5).
- FastFlow의 cait/deit 백본 미지원. 로컬 자산이 HF safetensors 디렉터리라 현재 `torch.load` 경로로 읽히지 않는다.
- PatchCore·PaDiM·DFM·DFKDE는 `runtime.amp: true`와 호환되지 않을 수 있다 (§7.2, §10.2).
- `weights_path=None`이면 기존 모델 전부 오류 없이 random-init backbone으로 빌드된다 (적대적 검토 Major).
- `_load_backbone_weights`가 여러 팩토리에 복제되어 있다 (§8.5, §15 참조 — 공통화는 별도 과제).
- Reverse Distillation `train.epochs: 200`은 RD4AD 논문 기준 잠정값이다.
- Reverse Distillation은 `input_size`와 `data.image_size` 불일치가 즉시 예외로 드러나지 않을 수 있다 (§9.3).
- CFA `train.epochs: 30`은 CFA 논문 기준 잠정값이다(§12).
- DFM `score_type="nll"` 경로는 upstream 자체 버그(§10.4)로 막혀 있고, 이 프로젝트의 pixel-metric 파이프라인과도 연결돼 있지 않다.
- `postprocess/__init__.py` re-export 정리.
- CFLOW `--resume`이 private decoder optimizer의 Adam 모멘텀을 보존하지 못한다 (§14.3, 적대적 검토 A2 Major).
- CFLOW의 `runtime.amp`/`train.grad_clip`이 private optimizer 경로에는 적용되지 않는다.
- **AUROC 지표 교체(§23.4) 이후 전 모델 수치 재측정 필요.** 점수·맵이 대략 17을 넘는 모델은 기존 기록이 부풀려져 있다. PaDiM(image 0.997→0.995)과 CS-Flow(pixel 0.500→0.646)는 확인됐고, PatchCore·CFLOW는 무영향으로 확인됐다. 나머지 11개 모델은 미확인이다. 3개 카테고리 정식 성능 측정 시 함께 갱신한다.
- **다단계 학습 모델의 스모크 epoch 수 기준 필요(§23.2).** DSR은 `--epochs 1`이면 정상 코드에서도 phase 1이 실행되지 않아 V-05를 통과하면서 결함이 숨었다. `/add-anomalib-model` 절차의 V-05에 "phase 경계를 넘는 epoch 수로 스모크한다"는 조건을 추가하는 것이 좋겠다.
- DSR `best image_auroc 0.667`(capsule, 10 epoch)은 논문 수치(~0.98)에 한참 못 미친다. phase 1이 7 epoch뿐이라 학습 예산이 절대 부족하다 — `--epochs 100` 이상으로 재확인 필요.
- CS-Flow는 256×256 입력에서 z 해상도가 8×8 / 4×4 / 2×2로 매우 거칠다. 지표를 고친 뒤에도 pixel 국소화 정밀도는 태생적으로 제한된다(upstream도 `input_size`를 데이터 쪽에서 받으므로 256이 틀린 값은 아니다).
- GANomaly는 MVTec에서 anomalib reference 자체가 평균 0.421로 0.5 미만이다(§23.1). 실사용 후보에서 제외하는 것이 타당하다.
- UniNet `weights_path=None` 시 source/target teacher가 조용히 랜덤 초기화된다 (적대적 검토 A5 Critical). PatchCore/PaDiM/DFM/DFKDE와 동일한 프로젝트 전역 패턴(§14 참조)이라 이번 세션에서는 수정하지 않음 — 별도 core 차원 과제로 이월.
- UniNet/Dinomaly/AnomalyDINO 3개 카테고리(bottle/carpet/capsule) 기준 정식 성능 비교 — **사용자 실행 대기**.
- Dinomaly `data.image_size=[392,392]`는 upstream Resize(448)+CenterCrop(392)의 근사치다. `anomaly_default` transform에 center-crop 단계를 추가하면(공통 코드 변경, 2개 이상 모델에서 필요 확인 시) 더 정확히 재현 가능.
- Dinomaly `train.epochs: 100`, UniNet `train.epochs: 100`은 잠정값이다(upstream trainer_arguments가 epoch 대신 max_steps=5000 또는 빈 dict를 사용).

---

작성일: 2026-08-23 (최종 갱신 2026-08-24)
문서 상태: FastFlow·PatchCore·PaDiM·Reverse Distillation·DFM·DFKDE·CFA·CFLOW·FRE·U-Flow·CS-Flow·SuperSimpleNet·GANomaly·DRAEM·DSR·UniNet·Dinomaly·AnomalyDINO 추가 산출물 (anomalib `091ca6a` 기준) + 사용자 실학습 결함 3건 수정 기록(§23) + Tier 3 적대적 검토 A5 결함 2건 수정 기록(§24~26)
