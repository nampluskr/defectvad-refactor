# CLI 사용법

이 저장소의 모든 실행은 `python -m src <command>` 하나로 통한다. 학습용 스크립트, 평가용 스크립트가 따로 없다.

## 1. 왜 `python -m src`인가

`src/` 폴더가 하나의 Python 패키지이고, 그 안의 `src/__main__.py`가 패키지를 직접 실행했을 때의 진입점이다. `python train.py` 대신 `python -m src train`을 쓰는 이유는 두 가지다.

1. `src.core`, `src.tasks` 같은 내부 import가 항상 같은 방식으로 해석된다. 스크립트를 직접 실행하면 실행 위치에 따라 import가 깨진다.
2. 어떤 명령을 쓰든 네트워크 차단 가드가 먼저 켜진다. `src/__main__.py:7`에서 torch를 포함한 무거운 모듈보다 먼저 `enable_offline_guard()`를 호출한다.

**반드시 저장소 루트에서 실행한다.** config 경로가 `configs/...` 같은 상대 경로로 쓰이기 때문이다.

## 2. 사전 준비

### 2.1. 환경 활성화

```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260820_defectvad-refactor
```

### 2.2. 로컬 경로 설정

데이터셋과 백본 가중치 위치는 머신마다 다르므로 config에 직접 쓰지 않고 `${paths.dataset_root}`, `${paths.backbone_root}` placeholder로 참조한다. 개발 머신 기본값은 `/mnt/d/datasets`, `/mnt/d/backbones`이며, 다른 값이 필요하면 `configs/local.yaml`을 만든다.

```bash
cp configs/local.example.yaml configs/local.yaml
```

`configs/local.yaml`은 gitignore 대상이라 커밋되지 않는다. 우선순위는 높은 쪽이 이긴다.

| 순위 | 방법 |
|---|---|
| 1 | `--set paths.dataset_root=...` |
| 2 | 환경변수 `DATASET_DIR` / `BACKBONE_DIR` |
| 3 | `configs/local.yaml` |
| 4 | 각 task `_base.yaml`의 `paths` 블록 |

### 2.3. 자산 점검

학습 전에 필요한 파일이 실제로 있는지부터 확인한다.

```bash
python -m src check-assets
```

`configs/assets.yaml`에 나열된 데이터셋 폴더와 백본 `.pth`를 하나씩 검사해 `OK` / `MISSING`과 크기를 표로 출력한다. 하나라도 없으면 종료 코드 1로 끝난다. **파일이 없어도 자동 다운로드하지 않는다.** 표에 찍힌 경로에 직접 파일을 놓아야 한다.

## 3. 명령어 한눈에 보기

| 명령 | 하는 일 | 필수 인자 |
|---|---|---|
| `check-assets` | 데이터셋·백본 존재 확인 | 없음 |
| `config` | config 최종 해석 결과 출력 | config 경로 |
| `train` | 학습 후 checkpoint 저장 | config 경로 |
| `evaluate` | checkpoint로 지표 측정 | config 경로, `--checkpoint` |
| `predict` | 이미지에 대해 추론 | config 경로, `--checkpoint`, `--input` |
| `benchmark` | 여러 모델을 같은 조건으로 비교 | benchmark 경로 |
| `leaderboard` | 기존 benchmark 결과 표 재생성 | benchmark 출력 폴더 |

전역 옵션 `--log-level`은 명령 앞에 둔다. 기본값은 `INFO`다.

```bash
python -m src --log-level DEBUG train configs/anomaly/stfpm.yaml
```

## 4. 전형적인 작업 흐름

```bash
# 1. 자산 확인
python -m src check-assets

# 2. 실제로 어떤 설정으로 돌아갈지 미리 확인 (학습 시작 전 dry-run)
python -m src config configs/anomaly/stfpm.yaml

# 3. 학습
python -m src train configs/anomaly/stfpm.yaml --set output.run_name=stfpm_bottle

# 4. test split으로 평가
python -m src evaluate configs/anomaly/stfpm.yaml \
    --checkpoint outputs/runs/anomaly/stfpm_bottle/checkpoints/best.pth \
    --split test

# 5. 개별 이미지 추론
python -m src predict configs/anomaly/stfpm.yaml \
    --checkpoint outputs/runs/anomaly/stfpm_bottle/checkpoints/best.pth \
    --input /mnt/d/datasets/mvtec/bottle/test/broken_large
```

## 5. 명령별 상세

### 5.1. `config` — 설정 확인

```bash
python -m src config configs/anomaly/stfpm.yaml
python -m src config configs/anomaly/stfpm.yaml --set data.params.category=capsule
```

`_base.yaml` 병합, `configs/local.yaml` 반영, `--set` 적용, `${paths.*}` 치환까지 끝난 최종 YAML을 그대로 출력한다. 학습이 오래 걸리는 만큼, 설정을 바꿨을 때는 먼저 이 명령으로 의도한 값이 들어갔는지 확인하는 편이 안전하다.

### 5.2. `train` — 학습

```bash
python -m src train configs/anomaly/stfpm.yaml --set output.run_name=stfpm_bottle
```

| 옵션 | 설명 |
|---|---|
| `--set KEY=VALUE` | config 값 덮어쓰기. 여러 번 반복 가능 |
| `--resume PATH` | checkpoint에서 이어서 학습 |

학습이 끝나면 best checkpoint를 다시 불러 valid를 한 번 더 평가한 결과를 `metrics_final.json`에 남긴다. 모델별 calibration이 학습 종료 시점에 적용되기 때문에, 학습 중 기록된 값이 아니라 이 값이 최종 성능이다.

중단된 학습을 이어서 하려면 다음과 같이 한다.

```bash
python -m src train configs/anomaly/stfpm.yaml \
    --set output.run_name=stfpm_bottle \
    --resume outputs/runs/anomaly/stfpm_bottle/checkpoints/last.pth
```

### 5.3. `evaluate` — 평가

```bash
python -m src evaluate configs/anomaly/stfpm.yaml \
    --checkpoint outputs/runs/anomaly/stfpm_bottle/checkpoints/best.pth \
    --split test
```

| 옵션 | 설명 |
|---|---|
| `--checkpoint PATH` | 필수 |
| `--split NAME` | 기본 `test`. `valid`도 가능 |

`test` split은 이 명령에서만 열린다. `train`은 `train`/`valid`만 접근하므로 학습 중 test 누수가 구조적으로 차단된다.

config의 `output.save_visualizations`가 true면 첫 배치에 대한 시각화 이미지가 `visualizations/`에 저장된다.

### 5.4. `predict` — 추론

```bash
python -m src predict configs/anomaly/stfpm.yaml \
    --checkpoint outputs/runs/anomaly/stfpm_bottle/checkpoints/best.pth \
    --input /mnt/d/datasets/mvtec/bottle/test/broken_large \
    --output outputs/predict_bottle
```

| 옵션 | 설명 |
|---|---|
| `--checkpoint PATH` | 필수 |
| `--input PATH` | 이미지 파일 하나 또는 이미지가 든 폴더 |
| `--output DIR` | 생략하면 `outputs/runs/<task>/...__predict`에 자동 생성 |

폴더를 주면 `.png`, `.jpg`, `.jpeg`를 이름순으로 모아 **한 배치로 한 번에** 처리한다. 파일이 아주 많은 폴더를 그대로 넘기면 메모리가 부족할 수 있다.

결과는 `predictions/predict.json`과 task별 산출물(anomaly의 경우 anomaly map)로 저장된다.

### 5.5. `benchmark` — 모델 비교

```bash
python -m src benchmark configs/benchmarks/anomaly_baseline.yaml
```

| 옵션 | 설명 |
|---|---|
| `--only NAME` | 특정 split만 실행 |
| `--overwrite` | 기존 결과 폴더를 덮어씀 |
| `--set KEY=VALUE` | 모든 split에 공통 적용 |

benchmark config는 하나의 `base` config를 두고 split마다 `override`로 모델만 바꾼다. 데이터·seed·image_size 같은 조건이 split 간에 동일한지 자동 검사하며, 어긋나면 control 위반으로 보고한다. 즉 "같은 조건에서 모델만 바꾼 비교"임을 도구가 보증한다.

`configs/benchmarks/anomaly_baseline.yaml`은 `custom_ae`, `stfpm`, `efficientad` 세 모델을 비교한다.

### 5.6. `leaderboard` — 표 재생성

```bash
python -m src leaderboard outputs/benchmarks/anomaly_baseline
```

이미 실행한 benchmark 결과 폴더를 다시 읽어 순위표만 다시 만든다. 학습은 다시 하지 않는다. 원래 실행에 있던 split 폴더가 사라졌으면 조용히 빼지 않고 오류를 낸다.

## 6. `--set` 사용법

`--set`은 config 파일을 고치지 않고 값 하나만 바꿀 때 쓴다. 키는 YAML 계층을 점으로 이어 쓰고, 값은 YAML 문법으로 해석된다.

```bash
# 카테고리 변경
--set data.params.category=capsule

# 에폭 수 변경 (정수로 해석됨)
--set train.epochs=1

# 리스트는 대괄호
--set data.image_size=[128,128]

# 리스트 원소는 인덱스로 지정
--set metrics.0.name=image_auroc

# null 지정
--set optim.scheduler=null

# 여러 개는 --set을 반복
--set data.params.category=carpet --set train.epochs=1 --set output.run_name=smoke
```

**중요: `--set`은 이미 있는 키만 덮어쓸 수 있다.** 없는 키를 지정하면 `references nonexistent key`라는 `ConfigError`가 난다. 새 항목을 추가하려면 config 파일 자체를 수정해야 한다. 오타를 조용히 무시하지 않기 위한 의도된 동작이다.

## 7. 출력물 구조

```text
outputs/
├── runs/<task_name>/<run_name>/
│   ├── config.resolved.yaml   # 실제 사용된 최종 설정 (재현용)
│   ├── env.json               # 실행 환경 정보
│   ├── train.log              # 로그
│   ├── metrics_epoch.csv      # epoch별 지표
│   ├── metrics_final.json     # 최종 지표
│   ├── checkpoints/
│   │   ├── best.pth
│   │   └── last.pth
│   └── visualizations/
└── benchmarks/<bench_name>/
    ├── splits/<split_name>/   # 위 run 폴더와 같은 구조
    ├── control_report.json    # 조건 동일성 검사 결과
    ├── leaderboard.csv         # 순위표
    └── leaderboard.md
```

`run_name`을 지정하지 않으면 `<config파일명>__<타임스탬프>` 형식으로 자동 생성된다. 나중에 찾기 쉽도록 `--set output.run_name=...`으로 직접 붙이는 편이 좋다.

**같은 `run_name`을 다시 쓰면 이전 결과 위에 덮어쓴다.** 재실행할 때는 이름을 바꾸거나 이전 폴더를 먼저 옮긴다.

## 8. 자주 겪는 상황

| 증상 | 원인과 해결 |
|---|---|
| `check-assets`에서 `MISSING` | 표에 찍힌 경로에 파일이 없다. 파일을 놓거나 `configs/local.yaml`의 경로를 고친다 |
| `ConfigError: paths.dataset_root does not exist` | `paths.dataset_root`/`paths.backbone_root` 자체가 존재하지 않는 경로다. 메시지가 안내하는 대로 `--set`, 환경변수, `configs/local.yaml` 중 하나로 고친다 |
| `ConfigError: ... nonexistent key` | `--set` 키 오타이거나 config에 없는 키다. `python -m src config`로 실제 키를 확인한다 |
| `OfflineViolationError` | 코드가 네트워크에 접근하려 했다. pretrained 가중치를 로컬 `.pth`로 지정했는지 확인한다 |
| `LocalAssetError` | 지정한 백본 `.pth`가 없다. 랜덤 초기화로 조용히 넘어가지 않고 의도적으로 실패시킨다 |
| CUDA out of memory | `--set data.batch_size=4`로 줄이거나 `--set data.image_size=[128,128]` |
| 빠른 동작 확인만 하고 싶다 | `--set train.epochs=1 --set data.batch_size=2 --set output.run_name=smoke` |

## 9. 참고

- 명령·인자 정의: `src/cli/parser.py`
- 각 명령의 실제 동작: `src/cli/commands.py`
- 설정 항목의 계약: `docs/dev/v0.1/SPEC.md`
