# CLI 사용 가이드 (CLI Usage Guide)

이 문서는 `cv_boilerplate` 프레임워크의 직접 실행 스크립트 진입점(`scripts/`) 및 3단계 CLI 인자 체계를 활용하는 방법을 설명한다.

---

## 1. CLI 아키텍처 및 3단계 인자 체계

v0.2부터 기존의 `python -m src <command>` 패키지 호출 방식을 완전히 폐기하고, 사용자가 직관적으로 제어할 수 있는 **스크립트 직접 실행 형태**로 전면 전환했다.

모든 실행 스크립트는 아래 3단계 계층의 인자 체계를 공통으로 지원한다.

```text
[CLI 3단계 인자 계층]
├── 1. 1급 표준 플래그 (First-class Flags)
│   ├── --data (-d), --model (-m), --config (-c) : 구성 파일 경로
│   ├── --epochs (-e), --batch-size (-b) : 핵심 학습 파라미터
│   ├── --output-dir (-o), --run-name : 산출물 경로 및 실험 식별자
│   ├── --checkpoint, --resume : 가중치 로드 및 학습 재개
│   ├── --split (-s), --input (-i) : 평가 split 및 추론 대상
│   └── --device, --seed, --print-config : 환경 제어 및 dry-run
│
├── 2. 동적 Selector (--data.<key>, --model.<key>)
│   └── YAML의 `selectors:` 블록에 선언된 키를 동적으로 파싱하여 연관 설정 동시 주입
│       (예: --data.category bottle, --model.backbone resnet50)
│
└── 3. 범용 오버라이드 (--set <dotted.key>=<value>)
    └── YAML 구조 내 임의의 기존 키를 점 표기로 직접 수정
        (예: --set runtime.device=cpu, --set train.monitor.mode=max)
```

---

## 2. 사전 준비 및 환경 설정

### 2.1. Conda 가상환경 활성화
```bash
conda activate pytorch_env
cd /mnt/d/projects/nampluskr/00_review/260820_defectvad-refactor
```

### 2.2. 로컬 머신 경로 설정 (`configs/local.yaml`)
데이터셋과 백본 가중치 위치는 머신마다 다르므로 config에 하드코딩하지 않고 `${paths.dataset_root}`, `${paths.backbone_root}` placeholder로 참조한다.

```bash
# 템플릿 복사 후 로컬 경로 지정
cp configs/local.example.yaml configs/local.yaml
```

경로 우선순위:
1. `--set paths.dataset_root=...` (CLI 최우선)
2. 환경변수 `DATASET_DIR` / `BACKBONE_DIR`
3. `configs/local.yaml` (로컬 머신 SSOT)
4. Task별 `_base.yaml`의 기본값

---

## 3. 명령어 한눈에 보기

| 스크립트 | 주요 역할 | 핵심 필수/주요 인자 |
|---|---|---|
| `scripts/train.py` | 모델 학습 및 체크포인트 저장 | `--data`, `--model`, `--output-dir`, `--run-name` |
| `scripts/evaluate.py` | 체크포인트 기반 split 지표 산출 | `--data`, `--model`, `--checkpoint`, `--split` |
| `scripts/predict.py` | 이미지 또는 폴더 대상 추론/시각화 | `--model`, `--checkpoint`, `--input` |
| `scripts/generate_splits.py` | 데이터셋 분할 파일(train/valid/test JSON) 생성 | `--data`, `--category`, `--out-dir` |
| `scripts/run_batch.py` | 다중 매트릭스 일괄 실행 | `--config`, `--mode` |
| `scripts/check_assets.py`| 로컬 데이터셋/가중치 무결성 점검 | `--config` (선택) |
| `scripts/report.py` | 실행 결과 지표 취합 및 리더보드 출력 | `--dir` |

---

## 4. 상세 사용법 및 예시

### 4.1. `scripts/train.py` — 모델 학습

Data Config와 Model Config를 직교 결합하여 모델을 학습한다.

#### (1) MVTec AD 학습 예시
```bash
# 1. STFPM + ResNet-18 (bottle 카테고리)
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet18 \
  --epochs 100 \
  --batch-size 16 \
  --output-dir outputs \
  --run-name stfpm_resnet18_bottle

# 2. EfficientAD Medium (grid 카테고리, batch_size=1 자동 적용)
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml --data.category grid \
  --model configs/anomaly/models/efficientad.yaml --model.size medium \
  --epochs 70 \
  --output-dir outputs \
  --run-name effad_medium_grid
```

#### (2) BTAD 및 VisA 데이터셋 학습 예시
```bash
# 1. BTAD 01 카테고리 학습
python scripts/train.py \
  --data configs/anomaly/data/btad.yaml --data.category 01 \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet18 \
  --epochs 50 \
  --batch-size 8 \
  --output-dir outputs \
  --run-name stfpm_btad_01

# 2. VisA candle 카테고리 학습
python scripts/train.py \
  --data configs/anomaly/data/visa.yaml --data.category candle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet18 \
  --epochs 50 \
  --batch-size 16 \
  --output-dir outputs \
  --run-name stfpm_visa_candle
```

#### (3) 중단된 학습 재개 (`--resume`)
```bash
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml \
  --resume outputs/stfpm_resnet18_bottle/checkpoints/last.pth
```

---

### 4.2. `scripts/evaluate.py` — 모델 평가

학습된 체크포인트를 로드하여 지정된 split(기본값: `test`)에 대해 Image AUROC, Pixel AUROC를 산출한다.

```bash
# 1. MVTec bottle 테스트셋 평가
python scripts/evaluate.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet18 \
  --checkpoint outputs/stfpm_resnet18_bottle/checkpoints/best.pth \
  --split test

# 2. BTAD 01 카테고리 테스트셋 평가
python scripts/evaluate.py \
  --data configs/anomaly/data/btad.yaml --data.category 01 \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet18 \
  --checkpoint outputs/stfpm_btad_01/checkpoints/best.pth \
  --split test

# 3. VisA candle 카테고리 검증셋(valid) 평가
python scripts/evaluate.py \
  --data configs/anomaly/data/visa.yaml --data.category candle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet18 \
  --checkpoint outputs/stfpm_visa_candle/checkpoints/best.pth \
  --split valid \
  --output-dir outputs/eval_results/visa_candle_valid
```

---

### 4.3. `scripts/predict.py` — 단일 / 폴더 추론 및 히트맵 시각화

단일 이미지 파일 또는 이미지들이 들어 있는 디렉터리에 대해 이상 탐지 추론을 수행하고, `predictions.json` 및 `visualizations/` 디렉터리에 `[원본 | 히트맵 | 오버레이]` 3열 합성 시각화 이미지를 저장한다.

```bash
# 1. 이미지 디렉터리 일괄 추론 및 시각화 저장
python scripts/predict.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet18 \
  --checkpoint outputs/stfpm_resnet18_bottle/checkpoints/best.pth \
  --input /mnt/d/datasets/mvtec/bottle/test/broken_large \
  --output-dir outputs/predictions/bottle_broken

# 2. 단일 이미지 대상 추론
python scripts/predict.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet18 \
  --checkpoint outputs/stfpm_resnet18_bottle/checkpoints/best.pth \
  --input /mnt/d/datasets/mvtec/bottle/test/broken_large/000.png \
  --output-dir outputs/predictions/single_sample
```

---

### 4.4. `scripts/generate_splits.py` — 데이터셋 분할 파일 생성

데이터셋 설정(`--data`)과 카테고리(`--category`)를 기반으로 각 데이터셋 클래스에 내장된 분할 알고리즘을 호출하여 해당 Task의 분할 디렉터리(예: `configs/anomaly/splits/`)에 Train/Valid/Test JSON 파일을 생성한다. (`--out-dir`를 지정하지 않으면 설정 파일의 `data.split.path` 경로를 기반으로 자동 저장됨)

```bash
# 1. MVTec AD 전체 카테고리 분할 생성 (configs/anomaly/splits/mvtec_*.json 저장)
python scripts/generate_splits.py --data configs/anomaly/data/mvtec.yaml --category all

# 2. BTAD 전체 카테고리 분할 생성 (configs/anomaly/splits/btad_*.json 저장)
python scripts/generate_splits.py --data configs/anomaly/data/btad.yaml --category all

# 3. VisA 특정 카테고리 분할 생성 (configs/anomaly/splits/visa_candle.json 저장)
python scripts/generate_splits.py --data configs/anomaly/data/visa.yaml --category candle
```

> [!NOTE]
> 생성된 분할 파일은 각 Task 폴더 하위(예: [`configs/anomaly/splits/`](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/configs/anomaly/splits/))에 저장되며, 학습 및 평가 시 Selector(`--data.category <name>`)를 통해 자동으로 주입됩니다.

---

### 4.5. 설정 확인 (Dry-run / `--print-config`)

실제 학습/평가를 시작하기 전에 셀렉터와 오버라이드가 반영된 최종 YAML 구조를 미리 확인할 수 있다.

```bash
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet50 \
  --print-config
```

---

### 4.6. 범용 오버라이드 (`--set`)

`--set`은 YAML 구조 내의 임의의 기존 키를 점 표기법으로 직접 수정할 때 사용한다.

```bash
# 디바이스 CPU 변경 및 에폭 축소
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml \
  --set runtime.device=cpu \
  --set train.epochs=1 \
  --set output.run_name=smoke_test
```

> [!IMPORTANT]
> `--set`은 기존에 존재하는 키만 덮어쓸 수 있습니다. 오타로 인한 잘못된 키 주입을 방지하기 위해 존재하지 않는 키를 지정하면 `ConfigError`가 발생합니다.

---

## 5. 산출물 구조 (Output Artifacts)

실행이 완료되면 `--output-dir` 하위에 다음과 같은 구조로 산출물이 정리된다:

```text
outputs/<run_name>/
├── config.yaml               # 실험 재현을 위한 최종 병합/해석 설정 파일
├── train.log                 # 학습 진행 로그
├── metrics_history.csv       # 에폭별 Loss, AUROC, Learning Rate 기록 CSV
├── metrics_test.json         # evaluate.py 실행 시 최종 테스트 평가 지표 JSON
├── predictions.json          # predict.py 실행 시 추론 스코어 및 메타데이터 JSON
├── visualizations/           # predict.py 실행 시 [원본 | 히트맵 | 오버레이] 이미지
│   ├── sample_000.png
│   └── ...
└── checkpoints/
    ├── best.pth              # 모니터링 지표 기준 최고 성능 체크포인트
    └── last.pth              # 마지막 에폭 체크포인트 (RNG 및 Optimizer 상태 보존)
```

---

## 6. 자주 겪는 상황 및 트러블슈팅

| 증상 | 원인 | 해결 방법 |
|---|---|---|
| `OfflineViolationError` | 모델/라이브러리가 외부 인터넷 접근을 시도함 | 오프라인 가드가 작동한 정상 동작입니다. 백본 가중치가 로컬 경로(`${paths.backbone_root}`)에 있는지 확인합니다. |
| `LocalAssetError: ... not found` | 필요한 로컬 가중치 또는 데이터셋 폴더가 없음 | `configs/local.yaml`의 경로가 올바른지 확인하고 실제 파일이 존재하는지 점검합니다. |
| `ConfigError: references nonexistent key` | `--set` 키 오타 또는 정의되지 않은 키 참조 | `--print-config`로 실제 존재하는 키 이름을 확인합니다. |
| `SplitError: split 'train' and 'valid' overlap` | 데이터 분할 파일에 중복 샘플이 포함됨 | 데이터 누수 방지 가드가 작동한 것입니다. 분할 파일(`configs/<task>/splits/*.json`)을 점검합니다. |
| CUDA Out of Memory | GPU 메모리 부족 | `--batch-size`를 줄이거나 `--set data.image_size=[128,128]`로 해상도를 조절합니다. |
