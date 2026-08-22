# BRIEF (Antigravity 안) — 실행 진입점 재설계, Data·Model 분리 및 CLI 표준 인자 체계

이 문서는 v0.2 BRIEF의 **Antigravity(AGY) 작성안**이다. 사용자가 제시된 안들([BRIEF-CLAUDE.md](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/dev/v0.2/BRIEF-CLAUDE.md), [BRIEF-CODEX.md](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/dev/v0.2/BRIEF-CODEX.md))과 본 안을 비교·검토하여 최종 [BRIEF.md](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/dev/v0.2/BRIEF.md)를 확정한다.

폴더 구조와 설계 원칙의 세부 내용은 [docs/guides/structure.md](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/guides/structure.md)를 참조한다.

---

## 1. 프로젝트 폴더 구조 요약 (Directory Structure)

`configs/`, `scripts/`, `src/` 3개 영역으로 역할을 분리한다.

### 1.1. `configs/` 구조 (Task 중심 하위 폴더 분리형)

```text
configs/
├── anomaly/
│   ├── _base.yaml          # Anomaly task 공통 기본값 (runtime, metrics, optim 등)
│   ├── data/
│   │   ├── mvtec.yaml      # MVTec AD 데이터셋 정의 및 category selector
│   │   ├── btad.yaml       # (v0.3 확장)
│   │   └── visa.yaml       # (v0.3 확장)
│   ├── models/
│   │   ├── stfpm.yaml      # STFPM 모델 정의 및 backbone selector
│   │   └── efficientad.yaml# EfficientAD 모델 정의 및 model_size selector
│   └── batch/
│       └── mvtec_matrix.yaml # Anomaly task 다중 조건 배치 매니페스트
├── classification/
│   ├── _base.yaml
│   ├── data/
│   │   └── cifar10.yaml
│   ├── models/
│   │   └── resnet18.yaml
│   └── batch/
│       └── cifar10_matrix.yaml
├── splits/
├── assets.yaml
└── local.example.yaml
```

### 1.2. `scripts/` 구조 (실행 진입점)

```text
scripts/
├── train.py                # UC-001: 단일 조건 학습 진입점
├── evaluate.py             # UC-002: 단일 조건 평가 진입점
├── predict.py              # UC-003: 단일 조건 추론/시각화 진입점
├── batch.py                # UC-004: 다중 조건 일괄 실행 진입점 (배치 러너)
├── check_assets.py         # UC-005: 로컬 데이터셋/백본 가중치 점검 유틸리티
└── report.py               # UC-006: 실행 결과 비교표/리더보드 집계 유틸리티
```

### 1.3. `src/` 구조 (내부 제품 소스 코드)

```text
src/
├── core/                   # Task-Agnostic 공통 프레임워크 엔진 (cv_boilerplate)
├── tasks/                  # Task별 어댑터, 데이터로더, 메트릭 (tasks/anomaly/ 등)
├── batch/                  # 배치 실행 엔진 (매트릭스/케이스 전개, runner, summary)
├── cli/                    # CLI 파서 및 인자 정의 헬퍼
├── data/                   # 공통 데이터 유틸리티
└── upstream/               # SSOT anomalib 원본 모델 코드 (수정 금지)
```

---

## 2. CLI 인자 체계

CLI 인자는 역할에 따라 3단계로 구분된다.

```text
[CLI 인자 체계]
├── 1. 1급 표준 인자 (First-class Flags)
│   ├── --data (-d), --model (-m) : 구성 파일 경로
│   ├── --epochs (-e), --batch_size (-b) : 핵심 파라미터
│   ├── --output_dir (-o), --run_name : 산출물 저장소 및 식별자
│   ├── --checkpoint (-c), --resume : 가중치 로드 및 재개
│   ├── --input (-i), --split (-s) : 추론 입력 및 평가 대상 split
│   └── --device, --seed : 실행 환경 제어
│
├── 2. 동적 Selector (--data.<key>, --model.<key>)
│   └── YAML의 `selectors:` 블록에 선언된 키를 동적으로 파싱하여 연동 주입
│
└── 3. 범용 오버라이드 (--set <dotted.key>=<value>)
    └── YAML 내 임의의 기존 키를 점 표기로 직접 수정
```

---

## 3. Use Case별 상세 정의 및 CLI 사용법

### UC-001. 단일 조건 학습 (`scripts/train.py`)

Data Config와 Model Config를 합성하여 모델을 학습하고 체크포인트를 저장한다.

```bash
# 1. 기본 실행
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml \
  --model configs/anomaly/models/stfpm.yaml

# 2. 카테고리, 백본, 에폭 지정 실행
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet50 \
  --epochs 100 \
  --batch_size 16 \
  --output_dir outputs/anomaly_stfpm \
  --run_name bottle_res50_e100 \
  --seed 42

# 3. 중단된 학습 재개 (Resume)
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml \
  --model configs/anomaly/models/stfpm.yaml \
  --resume outputs/anomaly_stfpm/bottle_res50_e100/checkpoints/last.pth
```

- **경계 분리**: 학습 완료 후 valid/test 재평가나 별도 리포트 생성 없이 가중치 저장 후 종료 (평가는 UC-002에서 전담).

---

### UC-002. 단일 조건 평가 (`scripts/evaluate.py`)

학습된 체크포인트를 로드하여 지정된 split(기본: `test`)에 대해 metric(Image AUROC, Pixel AUROC 등)을 산출하고 `metrics_test.json`을 생성한다.

```bash
# 1. 기본 테스트 평가 (Test split)
python scripts/evaluate.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml \
  --checkpoint outputs/anomaly_stfpm/bottle_res50_e100/checkpoints/best.pth

# 2. 검증셋 평가 (Valid split) 및 배치 크기/출력 경로 지정
python scripts/evaluate.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml \
  --checkpoint outputs/anomaly_stfpm/bottle_res50_e100/checkpoints/best.pth \
  --split valid \
  --batch_size 32 \
  --output_dir outputs/eval_results/bottle_valid

# 3. 모델 변형(Backbone / Model Size)에 맞춘 평가
python scripts/evaluate.py \
  --data configs/anomaly/data/mvtec.yaml --data.category grid \
  --model configs/anomaly/models/efficientad.yaml --model.size medium \
  --checkpoint outputs/anomaly_effad/grid_medium/checkpoints/best.pth
```

- **입력**: `--data` (필수), `--model` (필수), `--checkpoint` (필수).
- **출력**: 지표 결과 파일 `metrics_test.json` (또는 `metrics_valid.json`), 산출물 요약 로그.

---

### UC-003. 다중 조건 평가 및 일괄 실행 (`scripts/batch.py`)

여러 카테고리와 모델 조합을 한 번에 일괄 평가/학습한다.

#### 방법 A: 배치 매니페스트 기반 실행 (`scripts/batch.py`)

배치 매니페스트 파일(`configs/anomaly/batch/mvtec_stfpm_grid.yaml`)에 정의된 모든 조합을 순회하며 일괄 평가를 수행한다.

```bash
# 전체 15개 카테고리 × 모델 4종 일괄 평가
python scripts/batch.py \
  --config configs/anomaly/batch/mvtec_stfpm_grid.yaml \
  --mode evaluate
```

- **배치 매니페스트 예시 (`configs/anomaly/batch/mvtec_matrix.yaml`)**:
  ```yaml
  meta:
    name: mvtec_benchmark_eval
    task: anomaly

  base_data: configs/anomaly/data/mvtec.yaml
  base_model: configs/anomaly/models/stfpm.yaml

  matrix:
    data:
      category: [bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper]
    model:
      backbone: [resnet18, resnet50]

  execution:
    checkpoint_pattern: "outputs/{task}_{model}_{data_category}/checkpoints/best.pth"
    split: test
    output_root: "outputs/batch_eval"
  ```

#### 방법 B: 쉘 루프를 활용한 다중 평가 스크립트

매니페스트 파일 없이 터미널 쉘 명령어로 다중 카테고리를 순회 평가할 수도 있다.

```bash
for cat in bottle grid leather tile; do
  python scripts/evaluate.py \
    --data configs/anomaly/data/mvtec.yaml --data.category $cat \
    --model configs/anomaly/models/stfpm.yaml \
    --checkpoint outputs/exp_${cat}_stfpm/checkpoints/best.pth \
    --output_dir outputs/eval_results/${cat}
done

# 결과 집계 리포트 생성
python scripts/report.py --dir outputs/eval_results/
```

- **내결함성**: 특정 카테고리의 체크포인트가 없거나 오류가 발생해도 중단되지 않고 다음 카테고리 평가를 계속 수행.

---

### UC-004. 단일 조건 추론 (`scripts/predict.py`)

단일 이미지 또는 디렉터리 내 이미지들에 대해 비지도 이상 탐지(Anomaly Map, Score, Mask)를 수행한다.

```bash
python scripts/predict.py \
  --model configs/anomaly/models/stfpm.yaml \
  --checkpoint outputs/anomaly_stfpm/bottle_res50_e100/checkpoints/best.pth \
  --input /mnt/d/datasets/mvtec/bottle/test/broken_large \
  --output_dir outputs/predictions/bottle_broken_large \
  --device cuda
```

---

### UC-005. 보조 유틸리티

- **자산 점검**: `python scripts/check_assets.py --config configs/anomaly/data/mvtec.yaml`
- **리더보드/결과 집계**: `python scripts/report.py --dir outputs/`

---

## 4. Selector 동작 원리 및 파라미터 연동

### 4.1. YAML 내 `selectors` 선언 예시

#### Data Config (`configs/anomaly/data/mvtec.yaml`)
```yaml
_base: configs/anomaly/_base.yaml

data:
  name: mvtec_anomaly
  root: ${paths.dataset_root}/mvtec
  image_size: [256, 256]
  batch_size: 8
  num_workers: 4
  params:
    category: bottle
  split:
    mode: file
    path: configs/splits/mvtec_bottle.json

selectors:
  category:
    target:
      - data.params.category: "{value}"
      - data.split.path: "configs/splits/mvtec_{value}.json"
    choices: [bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper]
    default: bottle
    help: "MVTec AD category name"
```

#### Model Config (`configs/anomaly/models/stfpm.yaml`)
```yaml
_base: configs/anomaly/_base.yaml

model:
  name: stfpm
  params:
    backbone_name: resnet18
    layers: [layer1, layer2, layer3]

selectors:
  backbone:
    target:
      - model.params.backbone_name: "{value}"
    choices: [resnet18, resnet50, wide_resnet50_2]
    default: resnet18
    help: "Feature extractor backbone"
```

---

## 5. 유효성 검증 및 충돌 방지 정책

1. **Task 일치성 검증**: Data Config의 `meta.task_name`과 Model Config의 `meta.task_name`이 일치하지 않는 경우 즉시 `ConfigError` 발생.
2. **Selector 유효성 검증**: 정의되지 않은 selector 지정 시 허용 목록(`choices`)과 함께 오류 안내.
3. **충돌 방지 우선순위**:
   - `CLI --set` > `CLI --<selector>` > `1급 CLI 플래그` > `YAML 구체 설정` > `YAML _base`
4. **오프라인 원칙 준수 (원칙 3)**: 로컬 데이터셋 또는 백본 파일 부재 시 자동 다운로드하지 않고 즉시 `ConfigError` 발생.

---

## 6. v0.2 범위 정의

### 포함 범위
1. 스크립트 직접 실행 진입점 신설: `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`, `scripts/batch.py`.
2. config 해석 엔진 `resolve_config()`에 selector 해석 및 derived key 합성 로직 구현.
3. core 엔진/어댑터 계약 확정: `core/engine.py`가 Anomaly task의 lifecycle hooks(`train_step`, `eval_step`, `on_fit_start`, `on_fit_end`, `configure_optimizers`)를 pure-PyTorch로 오케스트레이션.
4. MVTec AD 단일 학습(`train.py`) 파이프라인 완성.
5. 단일 평가(`evaluate.py`) 및 다중 평가(`batch.py --mode evaluate`) 파이프라인 완성.
6. Anomaly task (STFPM, EfficientAD) 및 MVTec AD 단일/다중 조건 검증.

---

작성일: 2026-08-22
문서 상태: Antigravity(AGY) 제안안 — 사용자 승인 및 비교 후 `BRIEF.md`로 확정
