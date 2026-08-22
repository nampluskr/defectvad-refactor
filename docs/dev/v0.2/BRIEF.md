# BRIEF — 범용 CV 실행 프레임워크(cv_boilerplate) 재설계 및 진입점/설정 체계 정립

## 1. 프로젝트의 본질 및 핵심 목적

이 워크스페이스의 본질적인 역할은 **"STFPM과 EfficientAD 모델을 이상 탐지(Anomaly Detection) 레퍼런스로 삼고, 향후 Classification, Detection, Segmentation까지 유연하게 확장 및 실행 가능한 범용 컴퓨터 비전 프레임워크(`cv_boilerplate`)를 재설계하는 것"**이다.

기존의 `python -m src <subcommand>` 형태의 패키지 호출 방식을 완전히 폐기하고, 사용자가 직관적으로 제어할 수 있는 **스크립트 직접 실행 형태**로 전면 전환한다.

```bash
# 단일 조건 학습/평가/추론
python scripts/train.py --data <data.yaml> --model <model.yaml> [옵션]
python scripts/evaluate.py --data <data.yaml> --model <model.yaml> --checkpoint <ckpt.pth> [옵션]
python scripts/predict.py --model <model.yaml> --checkpoint <ckpt.pth> --input <path> [옵션]

# 다중 조건 배치 일괄 실행
python scripts/run_batch.py --config <batch.yaml> --mode train|evaluate|predict|all
```

---

## 2. 3대 불변 원칙 (Project Rules)

이 프로젝트의 모든 설계와 구현은 아래의 3대 원칙을 최우선으로 준수한다.

1. **원칙 1 — anomalib 모델 코드는 SSOT이며 수정하지 않는다**:
   - `src/tasks/anomaly/models/` 하위의 anomalib 기반 순수 PyTorch 코드는 불변의 SSOT(단일 진실 공급원)이다. 별도의 `upstream/` 폴더 없이 `models/`가 직접 SSOT 역할을 수행한다.
   - 통합 과정에서 발생하는 모든 lifecycle 차이, 통계 계산, shape 변환은 모델 코드가 아니라 어댑터(`src/tasks/anomaly/adapters/`)와 프레임워크(`src/core/`)에서 해결한다.
2. **원칙 2 — Lightning은 절대 사용하지 않는다**:
   - 상위 프레임워크 없이 순수 PyTorch(`cv_boilerplate`)로만 학습·평가·추론 파이프라인을 구축한다.
   - 모든 모델은 `torch.nn.Module`을 상속하는 순수 PyTorch 클래스로 작성한다.
3. **원칙 3 — 데이터셋과 pretrained 가중치는 로컬 자산만 사용한다**:
   - 인터넷을 통한 자동 다운로드를 전면 차단하며, 로컬 경로 부재 시 즉시 `ConfigError`로 중단한다.

---

## 3. 멀티태스크 아키텍처 및 Task-Agnostic 원칙

`cv_boilerplate`는 Anomaly 외에도 향후 Classification, Segmentation, Detection task를 동일한 엔진으로 실행할 수 있어야 한다.

```text
                                [scripts/ (공통 CLI 진입점)]
                                             │
                                             ▼
                               [src/core/engine.py (Engine)]
                               (Task/모델에 대한 분기문 0개)
                                             │
                     ┌───────────────────────┼───────────────────────┐
                     ▼                       ▼                       ▼
           [src/tasks/anomaly/]    [src/tasks/classification/]  [src/tasks/segmentation/]
          - adapters/ (Anomaly)   - adapters/ (Classify)       - adapters/ (Segment)
          - datasets/ (MVTec)     - datasets/ (CIFAR)          - datasets/ (VOC)
          - models/stfpm/ (SSOT)  - models/resnet/             - models/deeplabv3/
          - models/efficientad/   - models/vit/                - models/unet/
          - metrics/ (AUROC)      - metrics/ (Accuracy)        - metrics/ (mIoU)
```

- **공통 엔진 (`src/core/engine.py`)**:
  - `Engine`은 Task 종류나 모델명을 전혀 알지 못하며, 공통 학습 루프에 Task명 분기(`if task == 'anomaly'`)를 두지 않는다.
- **어댑터 계층 (`adapters/`)**:
  - Task 고유의 데이터 구조, loss 계산, feature 추출, 사후 평가(threshold, 분위수 캘리브레이션 등)는 어댑터의 lifecycle hook(`train_step`, `eval_step`, `on_fit_start`, `on_fit_end` 등)으로 완전 격리 흡수한다.
- **순수 PyTorch 모델 패키지 (`models/<model_name>/`)**:
  - 각 모델은 `models/<model_name>/` 하위의 독립 패키지로 격리되며 `torch.nn.Module`을 직접 상속한다.

---

## 4. 디렉터리 구조 요약

상세한 폴더 구조와 각 파일의 역할 정의는 [docs/guides/structure.md](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/guides/structure.md)를 표준 참조로 한다.

```text
.
├── configs/                        # Task별 격리 설정 체계 (data/, models/, batch/)
│   ├── anomaly/                    # Anomaly task (mvtec.yaml, stfpm.yaml, mvtec_matrix.yaml)
│   ├── classification/             # Classification task (cifar10.yaml, resnet18.yaml)
│   ├── splits/                     # 데이터셋 분할 파일
│   ├── assets.yaml                 # 로컬 자산 레지스트리
│   └── local.example.yaml          # 로컬 경로 오버라이드 템플릿
│
├── scripts/                        # 사용자 직접 실행 스크립트
│   ├── train.py                    # 단일 조건 학습
│   ├── evaluate.py                 # 단일 조건 평가
│   ├── predict.py                  # 단일 조건 추론
│   ├── run_batch.py                # 다중 조건 일괄 실행 (배치 러너)
│   ├── check_assets.py             # 로컬 자산 점검
│   └── report.py                   # 지표 취합 리포트
│
└── src/                            # 제품 소스 코드
    ├── core/                       # Task-Agnostic 프레임워크 엔진 (engine, config, checkpoint 등)
    ├── tasks/                      # Task별 전면 하위 폴더 분리형 패키지
    │   └── <task_name>/            # (anomaly, classification, detection, segmentation 등)
    │       ├── adapters/           # Task 어댑터 (base.py, <model>.py)
    │       ├── datasets/           # 데이터셋 로더 (base.py, <dataset>.py)
    │       ├── models/             # [SSOT] 모델별 독립 패키지 (models/<model_name>/model.py - nn.Module)
    │       ├── transforms/         # 전처리 파이프라인 (default.py)
    │       ├── metrics/            # 평가 지표 (torchmetrics 기반)
    │       ├── losses/             # 손실 함수
    │       └── postprocess/        # 후처리 및 시각화 (visualizer.py, smoother.py)
    └── bench/                      # 배치 실행 엔진 (matrix.py, runner.py, control.py)
```

---

## 5. CLI 인자 계층 구조 및 Selector 메커니즘

CLI 인터페이스는 사용 편의성과 유연성을 모두 확보하기 위해 3단계 계층으로 구성된다.

```text
[CLI 인자 체계]
├── 1. 1급 표준 인자 (First-class Flags)
│   ├── --data (-d), --model (-m) : 구성 파일 경로 (필수/준필수)
│   ├── --epochs (-e), --batch-size (-b) : 핵심 학습 파라미터
│   ├── --output-dir (-o), --run-name : 산출물 경로 및 실험 식별자
│   ├── --checkpoint (-c), --resume : 가중치 로드 및 학습 재개
│   ├── --input (-i), --split (-s) : 추론 대상 및 평가 split 지정
│   └── --device, --seed : 실행 환경 제어
│
├── 2. 동적 Selector (--data.<key>, --model.<key>)
│   └── YAML의 `selectors:` 블록에 선언된 키를 동적으로 파싱하여 연관 설정 동시 주입
│
└── 3. 범용 오버라이드 (--set <dotted.key>=<value>)
    └── YAML 구조 내 임의의 기존 키를 점 표기로 직접 수정
```

---

## 6. Use Case별 상세 정의

### UC-001. 단일 조건 학습 (`scripts/train.py`)
Data Config와 Model Config를 합성하여 모델을 학습하고 체크포인트를 저장한다.
```bash
python scripts/train.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml --model.backbone resnet50 \
  --epochs 100 --batch-size 16 --output-dir outputs/anomaly_stfpm --run-name bottle_res50
```

### UC-002. 단일 조건 평가 (`scripts/evaluate.py`)
학습된 체크포인트를 로드하여 지정된 split에 대해 metric(Image/Pixel AUROC 등)을 산출하고 `metrics_test.json`을 저장한다.
```bash
python scripts/evaluate.py \
  --data configs/anomaly/data/mvtec.yaml --data.category bottle \
  --model configs/anomaly/models/stfpm.yaml \
  --checkpoint outputs/anomaly_stfpm/bottle_res50/checkpoints/best.pth \
  --split test
```

### UC-003. 다중 조건 평가 및 일괄 실행 (`scripts/run_batch.py`)
여러 카테고리와 모델 조합을 매니페스트 또는 쉘 루프로 순회하며 일괄 실행한다.
```bash
python scripts/run_batch.py \
  --config configs/anomaly/batch/mvtec_matrix.yaml \
  --mode evaluate
```

### UC-004. 단일 조건 추론 (`scripts/predict.py`)
단일 이미지 또는 디렉터리에 대해 이상 탐지(Anomaly Map, Heatmap, Mask)를 수행한다.
```bash
python scripts/predict.py \
  --model configs/anomaly/models/stfpm.yaml \
  --checkpoint outputs/anomaly_stfpm/bottle_res50/checkpoints/best.pth \
  --input /path/to/test/images \
  --output-dir outputs/predictions
```

### UC-005. 보조 유틸리티
- **로컬 자산 점검**: `python scripts/check_assets.py --config configs/anomaly/data/mvtec.yaml`
- **리더보드/결과 집계**: `python scripts/report.py --dir outputs/`

---

## 7. v0.2 범위 정의

### 포함 범위
1. 스크립트 직접 실행 진입점 신설: `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`, `scripts/run_batch.py`.
2. `python -m src` 구형 진입점 및 `src/cli/` 중복 실행 흐름 제거 및 통합.
3. `configs/<task>/data/`, `configs/<task>/models/`, `configs/<task>/batch/` 구조로 설정 파일 재배치.
4. `src/tasks/<task>/` 하위의 컴포넌트별 패키지화(`adapters/`, `datasets/`, `models/<model>/`, `transforms/`, `metrics/`, `losses/`, `postprocess/`).
5. `src/tasks/anomaly/models/`를 직접 SSOT로 지정하고 별도 `upstream/` 폴더 제거.
6. 모든 모델의 `models/<model_name>/` 독립 패키지화 및 순수 PyTorch `nn.Module` 표준화.
7. Data/Model Config 분리 및 동적 `selectors` 해석 엔진(`apply_selectors`) 구현.
8. 1급 표준 CLI 플래그 및 `--set` 오버라이드 체계 완성.
9. Anomaly task (STFPM, EfficientAD) 및 MVTec AD 단일/다중 조건 검증.

---

작성일: 2026-08-22
문서 상태: Approved BRIEF (v0.2 기준 문서) — 후속 문서는 이 문서를 바탕으로 파생
