# UPSTREAM-INVENTORY — 복사 원본 출처·라이선스 인벤토리

상위 문서: [SPEC.md](../SPEC.md) §3 · 절차: [MODEL-ADD.md](MODEL-ADD.md)

P6-T02 산출물이다. `src/tasks/anomaly/upstream/` 하위 전 파일에 대해 anomalib 원본 경로, commit, 라이선스, 허용된 변경, 이를 소비하는 adapter와 lifecycle hook, 필요한 로컬 자산을 한 표로 연결한다. v0.3·v0.4에서 모델을 늘릴 때 이 표에 행을 추가한다.

## 1. 출처와 라이선스

| 항목 | 값 |
|---|---|
| 저장소 | `anomalib` (open-edge-platform) |
| commit | `091ca6aca92c8d0e416394f79e52f5a3cea3db73` (`v2.3.0`, 2026-03-20) |
| 로컬 경로 | `/mnt/d/projects/clones/anomalib` (sparse checkout: `models/image/stfpm`, `models/image/efficient_ad`, `models/components`, `data`) |
| 라이선스 | Apache License 2.0 |
| 저작권 | Copyright (C) 2022-2025 Intel Corporation |

복사한 9개 파일 모두 파일 첫 두 줄에 `# Copyright (C) ... Intel Corporation` / `# SPDX-License-Identifier: Apache-2.0` 헤더를 원본 그대로 보존한다. Apache-2.0의 고지 의무는 이 헤더 보존과 본 인벤토리로 충족한다. 헤더를 지우거나 재작성하는 것은 CON-001 위반이자 라이선스 위반이다.

## 2. 파일 인벤토리

경로는 `src/tasks/anomaly/upstream/` 기준(대상)과 `/mnt/d/projects/clones/anomalib/` 기준(원본)이다.

| 대상 | 원본 | 줄 수 | 저작권 연도 | 허용 변경 |
|---|---|---|---|---|
| `stfpm/torch_model.py` | `src/anomalib/models/image/stfpm/torch_model.py` | 164 | 2022-2025 | import 3건 |
| `stfpm/loss.py` | `src/anomalib/models/image/stfpm/loss.py` | 136 | 2022-2025 | 없음 |
| `stfpm/anomaly_map.py` | `src/anomalib/models/image/stfpm/anomaly_map.py` | 169 | 2022-2025 | 없음 |
| `efficient_ad/torch_model.py` | `src/anomalib/models/image/efficient_ad/torch_model.py` | 711 | 2023-2025 | import 1건 |
| `components/feature_extractors/timm.py` | `src/anomalib/models/components/feature_extractors/timm.py` | 200 | 2022-2025 | 없음 |
| `components/feature_extractors/utils.py` | `src/anomalib/models/components/feature_extractors/utils.py` | 84 | 2022-2025 | 없음 |
| `components/data/generic.py` | `src/anomalib/data/dataclasses/generic.py` | 815 | 2024 | 없음 |
| `components/data/torch_base.py` | `src/anomalib/data/dataclasses/torch/base.py` | 196 | 2024-2025 | import 1건 |
| `components/tiler.py` | `src/anomalib/data/utils/tiler.py` | 472 | 2022-2025 | 없음 |

`__init__.py`와 `lightning_model.py`는 **의도적으로 복사하지 않았다**. 전자는 패키지 `__init__`이 Lightning 경로를 끌어오기 때문이고(SPEC §3), 후자는 CON-002 때문이다. `lightning_model.py`의 내용은 읽어서 config와 adapter로 옮겨 적었다(§4).

## 3. 허용된 변경 — import 경로 전량

CON-001이 허용하는 유일한 변경이다. 아래 5줄이 전부이며, 그 외 차이는 모든 파일에서 0라인이다(AC-001에서 원본 diff로 확인).

| 파일:행 | 변경 후 |
|---|---|
| `stfpm/torch_model.py:36` | `from src.tasks.anomaly.upstream.components.data.torch_base import InferenceBatch` |
| `stfpm/torch_model.py:37` | `from src.tasks.anomaly.upstream.components.feature_extractors.timm import TimmFeatureExtractor` |
| `stfpm/torch_model.py:42` | `from src.tasks.anomaly.upstream.components.tiler import Tiler` (`TYPE_CHECKING` 블록) |
| `efficient_ad/torch_model.py:43` | `from src.tasks.anomaly.upstream.components.data.torch_base import InferenceBatch` |
| `components/data/torch_base.py:21` | `from src.tasks.anomaly.upstream.components.data.generic import ImageT, _GenericBatch, _GenericItem` |

원본은 각각 `from anomalib.data import InferenceBatch`, `from anomalib.models.components.feature_extractors import TimmFeatureExtractor`, `from anomalib.data.utils.tiler import Tiler`, `from anomalib.data.dataclasses.generic import ...` 형태였다.

### 3.1 무결성 확인 기준값

Phase 완료 시 `git status -- src/tasks/anomaly/upstream/`로 무변경을 확인한다. 저장소 밖에서 검증해야 할 때(zip 배포본 등)는 아래 SHA-256 앞 16자리를 쓴다.

| 파일 | sha256 (앞 16자리) |
|---|---|
| `components/data/generic.py` | `eb3b12dd997b9c59` |
| `components/data/torch_base.py` | `d3bd8695988ab53e` |
| `components/feature_extractors/timm.py` | `018f16cbba5fb85e` |
| `components/feature_extractors/utils.py` | `68db2ea3d0dd3a3e` |
| `components/tiler.py` | `e73f43f0a3fb4499` |
| `efficient_ad/torch_model.py` | `e49fb8de5e393731` |
| `stfpm/anomaly_map.py` | `a9e0afe2fe7fa89e` |
| `stfpm/loss.py` | `307a52ec27fb2822` |
| `stfpm/torch_model.py` | `877a7abb2dbfab9c` |

```bash
cd src/tasks/anomaly/upstream && find . -name '*.py' | sort | xargs sha256sum
```

## 4. 모델별 연결

### 4.1 STFPM

| 항목 | 값 |
|---|---|
| upstream 파일 | `stfpm/torch_model.py`, `stfpm/loss.py`, `stfpm/anomaly_map.py` |
| 공유 components | `feature_extractors/timm.py`, `feature_extractors/utils.py`, `data/torch_base.py`, `data/generic.py`, `tiler.py` |
| adapter | `src/tasks/anomaly/adapters/stfpm.py#StfpmAdapter` (`ADAPTERS: "stfpm"`) |
| 모델 팩토리 | 같은 파일 `build_stfpm` (`MODELS: "stfpm_anomaly"`) |
| config | `configs/anomaly/stfpm.yaml` |
| override한 hook | `train_step`만 (`upstream STFPMLoss` 호출) |
| 로컬 자산 | `${paths.backbone_root}/resnet18-f37072fd.pth` (torchvision ImageNet resnet18) |
| 자산 주입 지점 | 팩토리. `timm.create_model`을 `pretrained=False`로 임시 치환 후 teacher feature extractor에만 로드 (SPEC §4.6 A안) |
| `lightning_model.py`에서 옮긴 것 | `configure_optimizers` → config의 SGD(lr 0.4, momentum 0.9, wd 0.001), scheduler 없음 |

### 4.2 EfficientAD

| 항목 | 값 |
|---|---|
| upstream 파일 | `efficient_ad/torch_model.py` (loss·anomaly map 포함) |
| 공유 components | `data/torch_base.py`, `data/generic.py` |
| adapter | `src/tasks/anomaly/adapters/efficientad.py#EfficientAdAdapter` (`ADAPTERS: "efficientad"`) |
| 모델 팩토리 | 같은 파일 `build_efficientad` (`MODELS: "efficientad_anomaly"`) |
| config | `configs/anomaly/efficientad.yaml` |
| override한 hook | `train_step`, `on_fit_start`, `on_validation_start`, `on_fit_end` |
| 로컬 자산 (가중치) | `${paths.backbone_root}/efficientad_pretrained_weights/pretrained_teacher_small.pth` |
| 로컬 자산 (auxiliary) | `${paths.dataset_root}/imagenette2/train` (ImageNette, `ImageFolder` 구조) |
| 자산 주입 지점 | 팩토리는 경로만 보관, `on_fit_start`가 `strict=True`로 로드 |
| `lightning_model.py`에서 옮긴 것 | `configure_optimizers` → config의 Adam(lr 1e-4, wd 1e-5) + StepLR(step_size 19, gamma 0.1, `train.epochs: 20`과 짝) / `on_train_start` → `on_fit_start`(teacher 가중치·auxiliary loader·채널 통계) / `on_validation_start` → 동명 hook(90%·99.5% 분위수) / `prepare_pretrained_model`의 다운로드 분기 → 제거하고 로컬 strict load로 대체 / batch size 1·normalization 미사용 제약 → config |

### 4.3 components 공유 관계

| components 파일 | 소비자 | 비고 |
|---|---|---|
| `feature_extractors/timm.py` | STFPM `torch_model.py` | `_map_layer_to_idx`가 `feature_extractors/utils.py`를 쓴다 |
| `feature_extractors/utils.py` | `feature_extractors/timm.py` | 간접 의존 |
| `data/torch_base.py` | STFPM·EfficientAD `torch_model.py` | `InferenceBatch`만 실제로 쓰인다. 나머지 클래스는 미사용이나 파일 단위 복사 원칙으로 보존 |
| `data/generic.py` | `data/torch_base.py` | `data/torch_base.py`의 유일한 내부 의존 |
| `tiler.py` | STFPM `torch_model.py` | `TYPE_CHECKING` 블록에서만 참조. 런타임 실행 경로에는 없다(`tiler=None`) |

## 5. 로컬 자산 원장

`configs/assets.yaml`이 SSOT이며, 아래는 v0.1 anomaly task가 실제로 요구하는 항목만 추린 것이다. 어느 것도 자동 다운로드하지 않는다(CON-003·CON-004). 새 머신에서는 `python -m src check-assets`로 누락을 확인한다.

| assets.yaml 키 | 개발 머신 경로 | 요구 모델 |
|---|---|---|
| `weights.resnet18` | `/mnt/d/backbones/resnet18-f37072fd.pth` | STFPM |
| `weights.efficientad_teacher_small` | `/mnt/d/backbones/efficientad_pretrained_weights/pretrained_teacher_small.pth` | EfficientAD |
| `datasets.mvtec_bottle` | `/mnt/d/datasets/mvtec/bottle` | 공통 |
| `datasets.imagenette2` | `/mnt/d/datasets/imagenette2` | EfficientAD |

`weights.resnet18`은 timm `resnet18`이 아니라 **torchvision 가중치**다. 키 이름은 호환되나 학습 레시피가 다르므로, reference 성능 차이 조사 시 PRD §5.4 "pretrained weight" 항목의 후보다(SPEC §4.6).

## 6. 갱신 규칙

- 새 모델 추가 시 §2에 파일 행, §3에 치환한 import 행, §3.1에 해시, §4에 모델 절, §5에 자산 행을 추가한다.
- anomalib commit을 올리면 §1의 commit과 §3.1의 해시를 전부 다시 산출하고, 전 파일을 재diff한다.
- `upstream/` 파일이 변경된 것이 발견되면 즉시 되돌린다. 변경이 필요해 보이면 adapter 또는 boilerplate에서 해결한다.

---

작성일: 2026-08-21
문서 상태: P6-T02 산출물 (anomalib `091ca6a` 기준)
