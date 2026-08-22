# BRIEF (Claude 안) — train CLI 명령어 형식과 config 파일 형식

이 문서는 v0.2 BRIEF의 **Claude Code 작성안**이다. Codex가 별도 안을 작성하며, 사용자가 두 안을 비교해 `BRIEF.md`로 확정한다. 기존 `docs/dev/v0.2/BRIEF.md`는 그대로 둔다.

이번 문서의 목적은 두 가지를 확정하는 것이다.

1. **CLI 명령어 형식** — `train`을 기준으로 정한다. `evaluate`·`predict`는 같은 형식에서 실행 단계만 바뀌므로 따로 정의하지 않는다.
2. **config 파일 형식** — dataset과 model을 분리한 조합형 구조.

## 1. 설계 제약: cv_boilerplate는 task에 독립적이어야 한다

이 저장소의 `src/core/`는 anomaly를 포함해 5개 task(classification, detection, segmentation, toy, anomaly)가 공유하는 상위 프레임워크 `cv_boilerplate`에서 왔다. **CLI와 공통 코드는 어떤 task도 알아서는 안 된다.**

이 제약이 이번 형식 결정 전체를 지배한다. 구체적으로 다음을 금지한다.

| 금지 | 이유 |
|---|---|
| `--data mvtec` 처럼 **이름**을 받는 것 | CLI가 `configs/data/<name>.yaml` 같은 디렉터리 관례와 유효한 이름 목록을 알아야 한다 |
| `--model.backbone` → `model.params.backbone_name` 별칭 테이블 | CLI가 특정 모델의 파라미터 키 이름을 알아야 한다 |
| `--data.category`가 split 파일까지 바꾸는 것 | CLI가 MVTec의 파일명 관례를 알아야 한다 |

세 가지 모두 **경로를 직접 받고, 키는 일반 규칙으로 풀고, 파일 간 연동은 config가 선언**하는 방식으로 해결한다.

## 2. 배경

대표 시나리오는 "MVTec 4개 카테고리(bottle, grid, leather, tile) × 모델 4종(STFPM resnet18/resnet50, EfficientAD small/medium)을 epoch 100으로 학습"이다. 16조합이며, 이 시나리오가 형식 결정의 기준이다.

현재 인터페이스로 막히는 지점은 네 곳이다.

| 문제 | 근거 |
|---|---|
| dataset과 model이 한 config에 뭉쳐 있다 | 각 task의 `_base.yaml`이 data·model·optim·train을 모두 담아, 조합마다 파일이 필요하다 |
| 모델 변형을 CLI로 고를 수 없다 | `--set`은 config에 있는 키만 덮어쓰는데, STFPM backbone과 EfficientAD `model_size`는 yaml에 없고 팩터리 기본 인자로만 존재한다 |
| 카테고리를 바꾸면 split 파일이 따라오지 않는다 | `data.params.category`와 `data.split.path`가 독립 값이라 실행 중 실패한다 |
| epoch을 바꾸면 scheduler가 어긋난다 | `efficientad.yaml`의 `step_size: 19`는 `int(0.95 × epochs)`인데 손으로 맞춰야 한다 |

네 문제 모두 **config 파일 형식에서 푼다.** CLI는 형식을 단순하게 유지한다.

## 3. CLI 명령어 형식

### 3.1. 단일 조건 — `scripts/train.py`

**dataset config와 model config를 각각 경로로 지정하고, 세부 설정은 점 표기로 얹는다.**

```bash
cd <workspace>
python scripts/train.py --data <path> --model <path> [옵션]
```

실제 사용례.

```bash
# 기본값으로
python scripts/train.py \
    --data  configs/anomaly/mvtec.yaml \
    --model configs/anomaly/stfpm.yaml

# 카테고리와 backbone 지정
python scripts/train.py \
    --data  configs/anomaly/mvtec.yaml --data.category bottle \
    --model configs/anomaly/stfpm.yaml --model.backbone_name resnet50

# 실제 시나리오
python scripts/train.py \
    --data  configs/anomaly/mvtec.yaml --data.category grid \
    --model configs/anomaly/efficientad_medium.yaml \
    --train.epochs 100

# 어떤 task든 같은 형식
python scripts/train.py \
    --data  configs/classification/oxford_pets.yaml \
    --model configs/classification/resnet50.yaml

# 자기완결 config 하나로
python scripts/train.py --config configs/toy/toy_cls.yaml

# 실행 전에 최종 config만 확인
python scripts/train.py --data ... --model ... --print_config
```

### 3.2. 인자 목록

`--data`, `--model`, `--config` 중 **최소 하나는 있어야 한다.** 나머지는 값을 덮어쓰는 수단이다.

| 인자 | 의미 |
|---|---|
| `--data PATH` | dataset config 경로 |
| `--model PATH` | model config 경로 |
| `--config PATH` | 자기완결 config 경로. 재현·실험용으로 위에 얹을 수도 있다 |
| `--<section>.<key> VALUE` | config 값 덮어쓰기. 해석 규칙은 §3.3 |
| `--set KEY=VALUE` | 점 표기 전체 경로로 덮어쓰기. 반복 가능 |
| `--resume PATH` | checkpoint에서 이어 학습 |
| `--print_config` | 최종 해석 결과만 출력하고 종료 |
| `--log-level LEVEL` | 기본 `INFO` |

### 3.3. `--<section>.<key>` 해석 규칙

**별칭 테이블을 두지 않는다.** 규칙 하나로 해석한다.

```text
--<section>.<key> VALUE
  1) 병합된 config에 <section>.<key>가 있으면       -> <section>.<key>
  2) 없고 <section>.params.<key>가 있으면           -> <section>.params.<key>
  3) 둘 다 없으면 오류 (두 후보 경로를 모두 제시)
```

`<section>`은 config의 최상위 섹션명 그대로다 — `data`, `model`, `train`, `runtime`, `output`, `optim`, `adapter`. CLI가 아는 고정 목록은 이 섹션명뿐이며 task 개념이 아니다.

| 입력 | 해석 결과 | 규칙 |
|---|---|---|
| `--data.batch_size 4` | `data.batch_size` | 1 |
| `--data.image_size "[128, 128]"` | `data.image_size` | 1 |
| `--data.category bottle` | `data.params.category` | 2 |
| `--model.backbone_name resnet50` | `model.params.backbone_name` | 2 |
| `--model.model_size medium` | `model.params.model_size` | 2 |
| `--train.epochs 100` | `train.epochs` | 1 |
| `--runtime.seed 7` | `runtime.seed` | 1 |
| `--model.backbon resnet50` | 오류 | 3 |

이 규칙이 성립하는 이유는 `data.params`와 `model.params`가 각각 dataset·model 팩터리로 넘어가는 **자유 dict**이기 때문이다. 규칙은 config의 구조만 알고 내용은 모른다.

형식 규칙 네 가지.

- **`--data`/`--model`은 파일을 고르고, `--data.*`/`--model.*`는 값을 덮어쓴다.** 이름 앞부분이 같아 짝이 보이지만 역할이 다르다.
- **해석은 config 병합 이후에 한다.** 그래야 어느 후보가 실제로 존재하는지 판정할 수 있고, `--set`의 "존재하는 키만" 오타 가드가 그대로 적용된다.
- **`--<section>.<key>`와 `--set`은 하나의 오버라이드 경로로 합류한다.** 전자는 해석 후 `KEY=VALUE` 문자열이 되어 `--set` 목록에 들어간다. 해석 규칙이 두 벌 생기지 않고 YAML 값 문법도 그대로 따라온다.
- **같은 키를 둘이 동시에 가리키면 오류로 멈춘다.** 조용히 한쪽을 이기게 하지 않는다.

anomalib은 `--trainer.max_epochs`를 쓰지만 여기서는 섹션명 그대로인 `--train.epochs`를 쓴다. `trainer` → `train` 매핑을 두면 그것부터가 예외이기 때문이다.

### 3.4. evaluate / predict

**같은 형식에서 실행 단계만 바뀐다.** 인자 체계, 해석 규칙, config 병합 순서가 전부 동일하다. 단계별로 필요한 인자가 몇 개 더 붙을 뿐이다.

```bash
python scripts/evaluate.py --data ... --model ... --checkpoint <path> [--split test]
python scripts/predict.py  --data ... --model ... --checkpoint <path> --input <file|dir>
```

단계별 책임 경계만 정해 둔다.

| 단계 | 하는 일 | 산출물 |
|---|---|---|
| `train` | 학습과 checkpoint 저장까지 | `checkpoints/{best,last}.pth`, `metrics_epoch.csv`, `train.log` |
| `evaluate` | checkpoint로 지표 산출 | `metrics_final.json`, `visualizations/` |
| `predict` | checkpoint로 추론 | `predictions/` |

`train`은 학습 후 valid 재평가를 하지 않는다. 현행 `src/cli/commands.py::train()`이 fit 이후 재평가까지 한 함수에서 처리하는 것과 다른 경계다.

### 3.5. 다중 조건 — `scripts/batch.py`

```bash
python scripts/batch.py --config configs/anomaly/batch/<name>.yaml --mode train
python scripts/batch.py --config configs/anomaly/batch/<name>.yaml --mode all
python scripts/batch.py --config configs/anomaly/batch/<name>.yaml --mode evaluate --only ead_small_grid
```

| 인자 | 의미 |
|---|---|
| `--config PATH` | 필수. 다중 조건 config 경로 |
| `--mode {train,evaluate,predict,all}` | 실행할 단계. `all`은 case마다 세 단계 연속 |
| `--only NAME[,NAME]` | 특정 case만 실행 |
| `--overwrite` | 기존 산출물 덮어쓰기 |
| `--set KEY=VALUE` | 모든 case에 공통 적용 |

`--mode`는 실행 **단계**를 고른다. anomalib에서 `--model`은 모델을 가리키므로 그 이름을 단계 선택에 쓰지 않는다. case 하나가 실패해도 나머지는 계속 진행한다.

### 3.6. 보조 명령과 이관표

```bash
python scripts/check_assets.py                          # 데이터셋·가중치 존재 확인
python scripts/make_leaderboard.py <batch_output_dir>   # 결과 표 재생성
python scripts/show_config.py --data ... --model ...    # config 미리보기
```

기존 7개 서브커맨드가 빠짐없이 이관된다.

| 기존 | 신규 |
|---|---|
| `train` / `evaluate` / `predict` | `scripts/train.py` / `evaluate.py` / `predict.py` |
| `benchmark` | `scripts/batch.py --mode all` |
| `leaderboard` | `scripts/report.py` |
| `check-assets` | `scripts/check_assets.py` |
| `config` | `scripts/show_config.py` 또는 각 스크립트의 `--print_config` |

### 3.7. 폴더 구조

진입점을 성격에 따라 두 폴더로 나눈다. **사용자가 치는 명령은 전부 `scripts/` 하나에 모은다.**

```text
scripts/                          # 사용자가 실행하는 것
├── train.py  evaluate.py  predict.py
├── batch.py  report.py
├── check_assets.py  show_config.py
└── generate_mvtec_splits.py  generate_oxford_pets_*.py   (기존)

tools/                            # CI·검증 게이트 (scripts/에서 이동)
└── check_engine_purity.py  check_reproducibility.py  check_split_integrity.py
```

분리 기준은 **누가 언제 실행하는가**다. `scripts/`는 사용자가 작업 중 직접 치는 명령, `tools/`는 커밋 전이나 Phase 검증에서 도는 게이트다. 현재 `scripts/`에는 둘이 섞여 있어 폴더를 열어도 실행할 것을 골라낼 수 없다.

다중 조건 실행을 별도 최상위 폴더(`batch/`)로 빼지 않는다. `batch.py`도 사용자가 치는 명령이라는 점에서 `train.py`와 성격이 같고, 결과 조회인 `report.py`까지 넣으면 폴더 이름과 내용이 어긋난다. 단일과 배치의 구분은 폴더가 아니라 파일 이름과 `--mode`가 드러낸다.

`scripts/`와 `tools/`에는 `__init__.py`를 두지 않는다. `src/`만 패키지로 유지한다.

## 4. config 파일 형식

### 4.1. CLI가 하는 일은 "순서 있는 파일 목록 병합"뿐

```python
paths = [p for p in (data_path, model_path, config_path) if p]   # 최소 1개
config = load_and_merge_base(paths[0])
for path in paths[1:]:
    config = deep_merge(config, load_and_merge_base(path))
config = apply_overrides(config, cli_overrides)   # --<section>.<key> + --set
config = apply_derived(config, pinned_keys)       # SS4.4
config = resolve_paths(config, ...)
config = interpolate(config)
```

`load_and_merge_base`와 `deep_merge`는 `src/core/config.py`에 **이미 있다.** 새 병합 엔진을 만들지 않는다. `--data`/`--model`/`--config`는 이 목록의 이름 붙은 자리일 뿐이며, 이름이 붙은 이유는 두 가지다 — 명령을 읽으면 무엇이 데이터고 무엇이 모델인지 보이고, `--data.*`/`--model.*` 접두사가 의미를 갖는다.

**병합 순서 (뒤가 앞을 이긴다)**

| 순서 | 계층 | 담는 것 |
|---|---|---|
| 1 | `--data <path>` | `meta.task_name`, `data`, `metrics`, `loss`, `train.monitor`, `derive` |
| 2 | `--model <path>` | `model`, `adapter`, `optim` **+ 그 모델이 요구하는 data 오버라이드** |
| 3 | `--config <path>` (선택) | 자기완결 config, 재현용 `config.resolved.yaml`, 실험용 조각 |
| 4 | CLI (`--<section>.<key>`, `--set`) | 사용자가 직접 지정한 값 |
| 5 | `derive` | 짝지어진 값 자동 계산 (§4.4) |

전 task 공통값(`paths`, `runtime`, `train`, `output`)은 `configs/_base.yaml`에 두고 data config가 `_base: ../_base.yaml`로 상속한다. 기존 `_base` 상속 메커니즘을 그대로 쓴다.

**2번이 1번을 이기는 것은 의도적이다.** dataset과 model은 완전히 직교하지 않는다. EfficientAD의 `data.batch_size: 1`과 `transform.normalize: false`는 논문이 정한 **모델의 요구사항**이지 데이터셋의 성질이 아니다. 모델 config가 data 값을 덮어쓸 수 있어야 이 제약을 모델 쪽에 적어 둔다.

이 우선순위가 조용히 동작하면 위험하므로 `--print_config`가 **각 값의 출처 파일을 함께 출력**한다. 현행 `resolve_paths`가 `config["paths"]["_source"]`에 경로 해석 출처를 남기는 방식을 확장한다.

`meta.task_name`은 data config가 선언한다. 별도 `--task` 인자를 두지 않는다.

### 4.2. 폴더 구조 — 기존 task별 디렉터리를 그대로 쓴다

경로로 지정하므로 `configs/data/`·`configs/models/` 같은 새 디렉터리 관례가 필요 없다. 현재의 task별 디렉터리 안에서 data config와 model config가 나란히 산다.

```text
configs/
├── _base.yaml                    # 신규. 전 task 공통: paths, runtime, train, output
├── anomaly/
│   ├── mvtec.yaml                # 신규. data config
│   ├── stfpm.yaml                # 기존. model config (backbone_name 명시 추가)
│   ├── stfpm_resnet50.yaml       # 신규
│   ├── efficientad.yaml          # 기존 (+ model_size 명시)
│   ├── efficientad_medium.yaml   # 신규
│   └── custom_ae.yaml            # 기존
├── classification/
│   ├── oxford_pets.yaml          # 신규. data config
│   ├── resnet50.yaml             # 기존. model config
│   └── efficientnet_b0.yaml  custom_cnn.yaml
├── detection/  segmentation/     # 같은 패턴
├── toy/                          # 자기완결 fixture. --config로 실행 (분리하지 않음)
├── batch/  splits/  assets.yaml  local.yaml
```

v0.3의 데이터셋 확장(BTAD, VisA)은 `configs/anomaly/`에 data config 파일 하나를 더하는 것으로 끝난다. 모델 config는 손대지 않는다.

### 4.3. 계층별 예시

**`configs/_base.yaml`** — 전 task 공통

```yaml
paths:
  dataset_root: /mnt/d/datasets
  backbone_root: /mnt/d/backbones

runtime:
  seed: 42
  device: cuda
  amp: false
  deterministic: warn
  allow_network: false

train:
  epochs: 5
  grad_clip: null
  log_interval: 10
  save_last: true

output:
  root: outputs
  run_name: null
  save_predictions: true
  save_visualizations: true
  max_visualizations: 16
```

**`configs/anomaly/mvtec.yaml`** — data config

```yaml
_base: ../_base.yaml

meta:
  task_name: anomaly

data:
  name: mvtec_anomaly
  root: ${paths.dataset_root}/mvtec
  params: {category: bottle}
  image_size: [256, 256]
  batch_size: 8
  num_workers: 4
  drop_last: false
  split:
    mode: file
    path: configs/splits/mvtec_bottle.json   # derive가 category에 맞춰 덮어쓴다
  transform:
    train: {name: anomaly_default, params: {}}
    eval:  {name: anomaly_default, params: {}}

loss: {name: none, params: {}}
metrics:
  - {name: image_auroc, params: {}}
  - {name: pixel_auroc, params: {}}

train:
  monitor: {metric: image_auroc, mode: max}

derive:
  data.split.path: {template: "configs/splits/mvtec_{data.params.category}.json"}
```

**`configs/anomaly/stfpm.yaml`** — model config

```yaml
model:
  name: stfpm_anomaly
  params:
    backbone_name: resnet18                                  # 신규 명시
    weights_path: ${paths.backbone_root}/resnet18-f37072fd.pth

adapter:
  name: stfpm
  params: {smooth_sigma: 4.0}

optim:
  optimizer: {name: sgd, params: {lr: 0.4, momentum: 0.9, weight_decay: 0.001}}
  scheduler: null
```

**model config에는 `_base:`가 없다.** data config 위에 얹히는 조각이며 단독으로 완결되지 않는다. 현행 model config들이 갖고 있는 `_base: _base.yaml` 줄은 제거해야 한다 — 남겨 두면 data를 통째로 끌고 와 data config를 덮어쓴다.

**`configs/anomaly/stfpm_resnet50.yaml`**

```yaml
_base: stfpm.yaml
model:
  params:
    backbone_name: resnet50
    weights_path: ${paths.backbone_root}/resnet50-0676ba61.pth
```

model 위의 model 상속은 유지한다. 이건 data를 끌어오지 않으므로 안전하다.

**`configs/anomaly/efficientad.yaml`** — data를 덮어쓰는 예

```yaml
model:
  name: efficientad_anomaly
  params:
    model_size: small                                        # 신규 명시
    weights_path: ${paths.backbone_root}/efficientad_pretrained_weights/pretrained_teacher_small.pth

adapter:
  name: efficientad
  params:
    smooth_sigma: 4.0
    auxiliary_root: ${paths.dataset_root}/imagenette2/train
    auxiliary_seed: 42

# 이 모델의 요구사항이므로 data 계층을 덮어쓴다 (§4.1)
data:
  batch_size: 1
  transform:
    train: {name: anomaly_default, params: {normalize: false}}
    eval:  {name: anomaly_default, params: {normalize: false}}

optim:
  optimizer: {name: adam, params: {lr: 0.0001, weight_decay: 0.00001}}
  scheduler: {name: step, params: {step_size: 19, gamma: 0.1}}

train:
  epochs: 20

derive:
  optim.scheduler.params.step_size: {from: train.epochs, scale: 0.95, cast: int}
  adapter.params.auxiliary_seed:    {from: runtime.seed}
```

### 4.4. `derive` 블록

§2의 세 번째·네 번째 문제를 한 수단으로 푼다. 문법은 `eval` 없이 두 형태로 제한한다.

```yaml
derive:
  <대상 키>: {template: "<문자열>{다른.키}<문자열>"}
  <대상 키>: {from: <다른.키>, scale: <실수>, cast: int}
```

규칙.

- **대상 키가 config에 이미 있어야 한다.** 쓰기가 `--set`과 같은 경로를 타므로 오타 가드가 그대로 적용된다.
- **사용자가 직접 지정한 키는 건너뛴다.** 사용자 의도가 derive보다 우선한다.
- **`derive` 블록도 병합 대상**이라 data config와 model config의 derive가 합쳐진다.
- **평가는 병합과 CLI 오버라이드가 모두 끝난 뒤 한 번**, 경로 치환 앞에서 한다.
- 평가가 끝나면 config에서 제거한다.

split derive를 **data config가 소유**하는 것이 §1의 제약과 직결된다. 데이터셋이 제 파일명 관례를 스스로 선언하므로 `src/cli/`도 `src/core/`도 MVTec을 모른 채 동작하고, v0.3의 BTAD·VisA도 같은 방식으로 확장된다.

`--train.epochs 100`을 주면 EfficientAD의 `step_size`가 95로 따라온다. 현행 yaml의 "두 값을 함께 바꾸라"는 주석 두 개가 사라진다.

### 4.5. 모델 변형 키를 yaml에 명시

§2의 두 번째 문제에 대한 답이자, §3.3 일반 규칙이 성립하는 전제다. 키가 config에 없으면 규칙 3번으로 오류가 나므로 `--model.backbone_name`이 통하려면 키가 실제로 있어야 한다.

`--set` 가드를 완화하는 대안은 기각한다. 모델 팩터리가 `**params`로 끝나므로 가드를 풀면 `--set model.params.backbon=resnet50` 같은 오타를 조용히 삼키고 엉뚱한 모델을 학습한다.

가중치 경로까지 함께 바뀌는 변형은 전용 config로 둔다. `--model.backbone_name`만 바꾸면 `weights_path`가 resnet18을 가리킨 채 남아 실패하므로, `--model.weights_path`와 짝지어 쓰는 것으로 문서화한다.

### 4.6. 다중 조건 config

case가 CLI와 같은 경로 값을 쓴다.

```yaml
name: mvtec4_anomaly

control:
  enabled: false      # 모델마다 optimizer와 batch_size가 달라야 하는 sweep
  exceptions: []

common:
  data: configs/anomaly/mvtec.yaml       # 모든 case가 같으면 여기서 한 번
  set:
    - train.epochs=100
    - runtime.seed=42

cases:
  - {name: stfpm_r18_bottle,   model: configs/anomaly/stfpm.yaml,              set: [data.params.category=bottle]}
  - {name: stfpm_r18_grid,     model: configs/anomaly/stfpm.yaml,              set: [data.params.category=grid]}
  - {name: stfpm_r18_leather,  model: configs/anomaly/stfpm.yaml,              set: [data.params.category=leather]}
  - {name: stfpm_r18_tile,     model: configs/anomaly/stfpm.yaml,              set: [data.params.category=tile]}
  - {name: stfpm_r50_bottle,   model: configs/anomaly/stfpm_resnet50.yaml,     set: [data.params.category=bottle]}
  - {name: stfpm_r50_grid,     model: configs/anomaly/stfpm_resnet50.yaml,     set: [data.params.category=grid]}
  - {name: stfpm_r50_leather,  model: configs/anomaly/stfpm_resnet50.yaml,     set: [data.params.category=leather]}
  - {name: stfpm_r50_tile,     model: configs/anomaly/stfpm_resnet50.yaml,     set: [data.params.category=tile]}
  - {name: ead_small_bottle,   model: configs/anomaly/efficientad.yaml,        set: [data.params.category=bottle]}
  - {name: ead_small_grid,     model: configs/anomaly/efficientad.yaml,        set: [data.params.category=grid]}
  - {name: ead_small_leather,  model: configs/anomaly/efficientad.yaml,        set: [data.params.category=leather]}
  - {name: ead_small_tile,     model: configs/anomaly/efficientad.yaml,        set: [data.params.category=tile]}
  - {name: ead_medium_bottle,  model: configs/anomaly/efficientad_medium.yaml, set: [data.params.category=bottle]}
  - {name: ead_medium_grid,    model: configs/anomaly/efficientad_medium.yaml, set: [data.params.category=grid]}
  - {name: ead_medium_leather, model: configs/anomaly/efficientad_medium.yaml, set: [data.params.category=leather]}
  - {name: ead_medium_tile,    model: configs/anomaly/efficientad_medium.yaml, set: [data.params.category=tile]}
```

스키마 요소.

| 키 | 의미 |
|---|---|
| `name` | 출력 디렉터리와 결과 표의 이름 |
| `control.enabled` | 모델 외 조건 동일성 검증 여부. 기본 `true` |
| `control.exceptions` | `enabled: true`일 때 승인된 차이 목록 |
| `common.data` / `common.model` / `common.config` | 모든 case의 기본 경로 |
| `common.set` | 모든 case에 먼저 적용되는 오버라이드 |
| `cases[].name` | case 식별자. 출력 디렉터리 이름이자 `--only`의 인자 |
| `cases[].data` / `.model` / `.config` | `--data`/`--model`/`--config`와 같은 경로 값. 생략하면 `common` |
| `cases[].set` | 그 case만의 오버라이드 |

형식 결정 네 가지.

**case가 CLI와 같은 어휘를 쓴다.** 배치에서 실패한 case를 단일 명령으로 재현할 때 값을 그대로 옮겨 적으면 된다. 해석도 **`scripts/train.py`가 쓰는 것과 같은 함수**를 쓴다. case 파일도 task를 모른다 — `data`/`model`/`config`가 전부 경로일 뿐이라 같은 스키마로 5개 task의 배치를 표현한다.

**모델별 설정이 자동으로 따라온다.** 현행 manifest는 `base` + `model:` override 구조라 모델 설정이 따라오지 않고, 그래서 지금 `configs/benchmarks/anomaly_baseline.yaml`은 STFPM을 adapter `anomaly` + adamw로 돌리고 있다. 새 형식에서는 `model: configs/anomaly/stfpm.yaml`이 SGD lr 0.4와 adapter `stfpm`을 함께 가져온다.

**조합을 자동 전개하지 않는다.** anomalib benchmark는 `grid: [bottle, cable, capsule]` 식으로 곱셈을 하지만 여기서는 16개 이름을 모두 적는다. `--only ead_medium_grid`가 그대로 통하고, case 이름이 출력 경로와 1:1로 대응하며, `grep`으로 찾을 수 있다. `common:`은 모든 case에 동일한 설정만 담고 case 수를 늘리지 않는다.

**`control.enabled`가 거짓이어도 비교 리포트는 기록한다.** 검증을 끄는 것이지 기록을 끄는 것이 아니다. 기본값은 `true`로 두어 v0.1이 확보한 동등성 보장을 잃지 않고, sweep 파일이 명시적으로 `false`를 선언하게 한다.

### 4.7. 출력 구조

```text
outputs/
├── runs/<task_name>/<run_name>/          # 단일 조건
│   ├── config.resolved.yaml              # 병합·오버라이드·derive가 끝난 최종 config
│   ├── train.log / evaluate.log / predict.log
│   ├── metrics_epoch.csv
│   ├── metrics_final.json                # evaluate 단계가 생성
│   ├── env.json
│   ├── checkpoints/{best,last}.pth
│   ├── visualizations/
│   └── predictions/
└── batch/<name>/                         # 다중 조건
    ├── cases/<case_name>/                # 위 run 폴더와 같은 구조
    ├── control_report.json
    ├── batch_report.json                 # case별 status, 단계별 소요, 오류
    ├── leaderboard.csv
    └── leaderboard.md
```

`run_name` 기본값은 **선택된 파일들의 stem을 잇는 방식**이다(`mvtec__stfpm__<타임스탬프>`). 파일명에서 뽑으므로 task 지식이 필요 없다.

## 5. 이관 범위 — 5개 task 전부

**5개 task를 모두 data/model 분리로 이관한다.** 경로 기반 선택자라 CLI 변경 없이 적용되고, 두 관례가 공존할 이유가 없다.

예상보다 싼 이유는 **모델 config가 이미 전 task에 model-only 파일로 존재하기 때문**이다. 예를 들어 `configs/classification/resnet50.yaml`은 `_base: _base.yaml` 한 줄과 `model` 블록뿐이다. 이관 작업은 두 가지다.

1. 각 task `_base.yaml`의 data 부분을 `<dataset>.yaml`로 떼어낸다 (anomaly, classification, detection, segmentation — 4개 파일 신설).
2. 각 model config에서 `_base: _base.yaml` 줄을 제거한다 (약 12개 파일).

`configs/toy/`는 자기완결 fixture이므로 분리하지 않고 `--config`로 실행한다.

**이관 중 값이 바뀌면 조용히 다른 학습이 된다.** 이관 전 각 task의 해석 결과를 저장해 두고 이관 후 결과와 diff하는 것을 필수 게이트로 삼는다. 특히 model config에서 `_base:` 줄을 빼먹으면 data를 통째로 끌고 와 data config를 덮어쓰므로, 이 diff가 그것을 잡는다.

## 6. 선행 조건

에이전트가 대신 수행하지 않는다(원칙 3).

1. `configs/splits/`에 bottle·capsule·carpet만 있다. **grid·leather·tile split 파일 생성**이 필요하다.
   ```bash
   python scripts/generate_mvtec_splits.py grid
   python scripts/generate_mvtec_splits.py leather
   python scripts/generate_mvtec_splits.py tile
   ```
2. **EfficientAD medium teacher 가중치**가 로컬에 없으면 medium 4 case를 범위에서 뺀다. `configs/assets.yaml`에 해당 항목이 아직 없다.
3. `configs/assets.yaml`의 데이터셋 항목이 `mvtec_bottle`뿐이라 새 카테고리를 커버하지 못한다.
4. EfficientAD medium의 `teacher_out_channels`가 384와 다른지 확인이 필요하다.

## 7. 남은 결정

- 모델 config가 data 값을 덮어쓸 때(§4.1) 이를 실행 로그로도 알릴지, `--print_config`의 출처 표기로만 남길지.
- `--<section>.<key>`에서 같은 이름이 최상위와 `params` 양쪽에 있을 때. 규칙 1번(최상위)이 이기되 경고를 남기는 것으로 충분한지.
- model config가 실수로 `data`/`metrics`/`loss` 최상위 키를 갖는지 `validate_config`가 점검할지. EfficientAD처럼 의도적으로 갖는 경우가 있어 단순 금지는 안 된다.

## 8. 원칙과 비범위

### 8.1. 원칙

- **원칙 1** — anomalib 모델 코드는 SSOT이며 수정하지 않는다. 이번 버전은 `upstream/`에 접근하지 않는다.
- **원칙 2** — Lightning을 사용하지 않는다. anomalib `Engine`·CLI도 import하지 않으며 **인자 명명 관례만** 참고한다.
- **원칙 3** — 데이터셋과 pretrained 가중치는 로컬 자산만 쓴다. 없으면 무엇이 어느 경로에 필요한지 알리고 대기한다.
- **task 독립성** — §1. CLI와 `src/core/`는 어떤 task도 알지 못한다. 이 원칙이 §3.3의 일반 규칙과 §4.1의 경로 병합을 강제한다.

`docs/refs/BRIEF.md` NG-01이 anomalib CLI orchestration을 비목표로 두었으므로, **관례는 빌리되 구현은 빌리지 않는다**가 지켜야 할 선이다.

### 8.2. 비범위

- 조합 자동 전개(grid)
- 병렬·GPU 스케줄링의 실제 구현, retry·timeout 정책
- 새 데이터셋(BTAD, VisA) 추가, 새 모델 추가 — v0.3. 다만 이번 config 형식이 그 추가를 파일 하나로 끝나게 만든다
- `configs/toy/`의 data/model 분리 — 자기완결 fixture로 유지
- `roi-corner-detection` 레거시 코드의 이식 — 인터페이스 형태만 참고한다
- monitor metric 포화 대응, 카테고리별 threshold — v0.3
- 실험 관리 도구(tensorboard, wandb) 도입

---

작성일: 2026-08-22
작성자: Claude Code
문서 상태: 비교용 작성안 — Codex 안과 비교해 `BRIEF.md`로 확정한 뒤 PRD 이하를 작성한다
