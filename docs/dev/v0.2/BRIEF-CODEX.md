# BRIEF (Codex 안) — Task 독립형 Data·Model 실행

이 문서는 v0.2의 비교안이다. `src/core`는 task·dataset·model 이름을 알지 않고, 사용자가 지정한 data config와 model config를 일반 규칙으로 합성한다. 현재 구현·검증 범위는 anomaly의 `train`이다.

## 1. CLI

### STFPM

```bash
python scripts/train.py \
  --data configs/anomaly/mvtec.yaml \
  --data.category bottle \
  --model configs/anomaly/stfpm.yaml \
  --model.backbone resnet50 \
  --set train.epochs=100
```

### EfficientAD

```bash
python scripts/train.py \
  --data configs/anomaly/mvtec.yaml \
  --data.category bottle \
  --model configs/anomaly/efficientad.yaml \
  --model.size small \
  --set train.epochs=100
```

### 다중 조건

```bash
python scripts/run_batch.py \
  --config configs/batch/mvtec_anomaly.yaml \
  --mode train
```

```bash
python scripts/run_batch.py \
  --config configs/<batch>.yaml \
  --mode train|evaluate|predict|all
```

- `--data`와 `--model`은 YAML 경로다.
- `--data.*`와 `--model.*`은 각 YAML의 `selectors`에서 동적으로 정의한다.
- 필수 selector 누락, 알 수 없는 selector, data/model task 불일치는 실행 전에 오류로 처리한다.
- selector와 `--set`이 동일하거나 상하위 관계인 config key를 변경하면 충돌 오류로 처리한다.
- v0.2에서는 `train`만 구현·검증한다.

## 2. Config 배치

```text
configs/
├── anomaly/
│   ├── _base.yaml
│   ├── mvtec.yaml
│   ├── stfpm.yaml
│   └── efficientad.yaml
└── batch/
    └── mvtec_anomaly.yaml
```

```text
_base < data fragment < model fragment < selectors < --set
```

두 fragment는 같은 `_base`를 사용하고, 합성 후 하나의 기존 config contract를 만든다. 공통 코드는 selector patch와 derived rule만 해석하며 고유 이름으로 분기하지 않는다.

### `configs/anomaly/_base.yaml`

```yaml
meta:
  task_name: anomaly

paths:
  dataset_root: /mnt/d/datasets
  backbone_root: /mnt/d/backbones

runtime:
  seed: 42
  device: cuda
  amp: false
  deterministic: warn
  allow_network: false

loss:
  name: none
  params: {}

metrics:
  - {name: image_auroc, params: {}}
  - {name: pixel_auroc, params: {}}

train:
  epochs: 5
  grad_clip: null
  monitor: {metric: image_auroc, mode: max}
  log_interval: 10
  save_last: true

output:
  root: outputs
  run_name: null
  save_predictions: true
  save_visualizations: true
  max_visualizations: 16
```

### `configs/anomaly/mvtec.yaml`

```yaml
_base: _base.yaml

data:
  name: mvtec_anomaly
  root: ${paths.dataset_root}/mvtec
  params: {}
  image_size: [256, 256]
  batch_size: 8
  num_workers: 4
  drop_last: false
  split:
    mode: file
  transform:
    train: {name: anomaly_default, params: {}}
    eval: {name: anomaly_default, params: {}}

selectors:
  category:
    required: true
    choices:
      bottle:
        data:
          params: {category: bottle}
          split: {path: configs/splits/mvtec_bottle.json}
      grid:
        data:
          params: {category: grid}
          split: {path: configs/splits/mvtec_grid.json}
      leather:
        data:
          params: {category: leather}
          split: {path: configs/splits/mvtec_leather.json}
      tile:
        data:
          params: {category: tile}
          split: {path: configs/splits/mvtec_tile.json}
```

### `configs/anomaly/stfpm.yaml`

```yaml
_base: _base.yaml

model:
  name: stfpm_anomaly
  params: {}

adapter:
  name: stfpm
  params: {}

optim:
  optimizer:
    name: sgd
    params: {lr: 0.4, momentum: 0.9, weight_decay: 0.001}
  scheduler: null

selectors:
  backbone:
    required: true
    choices:
      resnet18:
        model:
          params:
            backbone: resnet18
            weights_path: ${paths.backbone_root}/resnet18-f37072fd.pth
      resnet50:
        model:
          params:
            backbone: resnet50
            weights_path: ${paths.backbone_root}/resnet50-0676ba61.pth
```

### `configs/anomaly/efficientad.yaml`

```yaml
_base: _base.yaml

model:
  name: efficientad_anomaly
  params: {}

adapter:
  name: efficientad
  params:
    smooth_sigma: 4.0
    auxiliary_root: ${paths.dataset_root}/imagenette2/train
    auxiliary_seed: 42

data:
  batch_size: 1
  transform:
    train: {name: anomaly_default, params: {normalize: false}}
    eval: {name: anomaly_default, params: {normalize: false}}

optim:
  optimizer:
    name: adam
    params: {lr: 0.0001, weight_decay: 0.00001}
  scheduler:
    name: step
    params: {gamma: 0.1}

derived:
  - target: optim.scheduler.params.step_size
    source: train.epochs
    multiply: 0.95
    round: floor

selectors:
  size:
    required: true
    choices:
      small:
        model:
          params:
            model_size: small
            weights_path: ${paths.backbone_root}/efficientad_pretrained_weights/pretrained_teacher_small.pth
      medium:
        model:
          params:
            model_size: medium
            weights_path: ${paths.backbone_root}/efficientad_pretrained_weights/pretrained_teacher_medium.pth
```

## 3. Batch config

```yaml
schema_version: 1
name: mvtec_anomaly

defaults:
  data:
    config: configs/anomaly/mvtec.yaml
  train:
    epochs: 100

cases:
  - name: stfpm_resnet18_bottle
    data: {category: bottle}
    model: {config: configs/anomaly/stfpm.yaml, backbone: resnet18}

  - name: stfpm_resnet18_grid
    data: {category: grid}
    model: {config: configs/anomaly/stfpm.yaml, backbone: resnet18}

  - name: stfpm_resnet18_leather
    data: {category: leather}
    model: {config: configs/anomaly/stfpm.yaml, backbone: resnet18}

  - name: stfpm_resnet18_tile
    data: {category: tile}
    model: {config: configs/anomaly/stfpm.yaml, backbone: resnet18}

  - name: stfpm_resnet50_bottle
    data: {category: bottle}
    model: {config: configs/anomaly/stfpm.yaml, backbone: resnet50}

  - name: stfpm_resnet50_grid
    data: {category: grid}
    model: {config: configs/anomaly/stfpm.yaml, backbone: resnet50}

  - name: stfpm_resnet50_leather
    data: {category: leather}
    model: {config: configs/anomaly/stfpm.yaml, backbone: resnet50}

  - name: stfpm_resnet50_tile
    data: {category: tile}
    model: {config: configs/anomaly/stfpm.yaml, backbone: resnet50}

  - name: efficientad_small_bottle
    data: {category: bottle}
    model: {config: configs/anomaly/efficientad.yaml, size: small}

  - name: efficientad_small_grid
    data: {category: grid}
    model: {config: configs/anomaly/efficientad.yaml, size: small}

  - name: efficientad_small_leather
    data: {category: leather}
    model: {config: configs/anomaly/efficientad.yaml, size: small}

  - name: efficientad_small_tile
    data: {category: tile}
    model: {config: configs/anomaly/efficientad.yaml, size: small}

  - name: efficientad_medium_bottle
    data: {category: bottle}
    model: {config: configs/anomaly/efficientad.yaml, size: medium}

  - name: efficientad_medium_grid
    data: {category: grid}
    model: {config: configs/anomaly/efficientad.yaml, size: medium}

  - name: efficientad_medium_leather
    data: {category: leather}
    model: {config: configs/anomaly/efficientad.yaml, size: medium}

  - name: efficientad_medium_tile
    data: {category: tile}
    model: {config: configs/anomaly/efficientad.yaml, size: medium}
```

## 4. 공통 실행 계약

```text
parse data/model paths
  -> load both fragments and their base
  -> verify base identity and meta.task_name
  -> register fragment selectors
  -> apply selector patches
  -> reject conflicting --set keys
  -> apply --set
  -> resolve paths and derived values
  -> validate final config and local assets
  -> train
```

- 단일 실행과 batch 실행은 같은 config 합성기와 학습 함수를 사용한다.
- data/model fragment가 선언한 이름 외에는 공통 코드에 task·dataset·model 문자열을 두지 않는다.
- 자산을 자동으로 다운로드하지 않는다.
- case 실패 후에도 다음 case를 실행하고 성공·실패 상태를 기록한다.

## 5. 비범위

- `evaluate`, `predict`, `all` 구현 및 산출물 정의
- classification, detection, segmentation, toy config 마이그레이션
- 조합 자동 생성
- 최상위 `batch/` 폴더
- 병렬 실행, GPU scheduling, retry, timeout
- 새 모델, 새 데이터셋, Lightning, 실험 관리 도구
- `upstream/` 수정

---

작성일: 2026-08-22  
작성자: Codex  
문서 상태: 비교용 작성안
