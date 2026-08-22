# 프로젝트 폴더 구조 가이드 (Directory Structure)

이 문서는 이 저장소의 전체 디렉터리 구조와 각 폴더 및 파일의 역할, 설계 원칙을 정의한 가이드이다.

---

## 1. 핵심 설계 원칙

1. **Task-Agnostic 공통 계층 (`src/core/`)**: 범용 컴퓨터 비전 프레임워크인 `cv_boilerplate`의 핵심 엔진(`engine.py`, `config.py`, `checkpoint.py` 등)은 특정 task(anomaly, classification 등)의 도메인 지식을 갖지 않는다.
2. **Task 격리형 Config 체계 (`configs/<task>/`)**: 각 Task별로 `data/`, `models/`, `batch/`를 하위에 격리하여 데이터셋과 모델을 $M \times N$으로 직교 조합한다.
3. **완전 폴더 모듈화 Task 구조 (`src/tasks/<task>/`)**: 모든 Task 컴포넌트(`adapters/`, `datasets/`, `models/<model>/`, `transforms/`, `metrics/`, `losses/`, `postprocess/`)를 독립 패키지 폴더로 격리한다.
4. **순수 PyTorch `nn.Module` 모델 자립성 및 SSOT (`src/tasks/<task>/models/<model>/`)**:
   - 모든 모델은 `models/<model_name>/` 하위의 독립 패키지로 위치하며 `torch.nn.Module`을 상속한다.
   - **원칙 1 적용**: `src/tasks/anomaly/models/` 하위의 anomalib 기반 순수 PyTorch 코드는 SSOT(단일 진실 공급원)로 취급하며, import 경로 외에는 수정하지 않는다. 별도의 `upstream/` 폴더를 두지 않고 `models/`가 직접 SSOT 역할을 수행한다.
5. **직접 실행형 스크립트 진입점 (`scripts/`)**: 패키지 모듈 호출 대신 사용자가 터미널에서 직관적으로 실행할 수 있는 표준 스크립트를 제공한다.

---

## 2. 전체 디렉터리 개요

```text
.
├── configs/                # Task별 설정 파일 (Data, Model, Batch, Base)
├── scripts/                # 사용자 직접 실행 스크립트 (학습, 평가, 추론, 배치)
├── src/                    # 제품 소스 코드 (프레임워크 코어, 태스크 패키지)
├── docs/                   # 개발 및 사용자 문서
│   ├── dev/                # 버전별 개발 문서 체인 (BRIEF, PRD, SPEC, PLAN 등)
│   ├── guides/             # 사용자 및 아키텍처 가이드
│   └── refs/               # 이전 분석 및 참조 문서 (읽기 전용)
├── AGENTS.md               # AI 에이전트 작업 지침 및 제약사항
└── README.md
```

---

## 3. 세부 영역별 구조

### 3.1. `configs/` (설정 영역)

Task 간 설정을 격리하고 Data, Model, Batch 매니페스트를 직교적으로 조합할 수 있도록 `configs/<task>/` 하위에 `data/`, `models/`, `batch/`를 각각 위치시킨다.

```text
configs/
├── anomaly/                        # Anomaly Detection Task 전용 설정
│   ├── _base.yaml                  # Task 공통 기본값 (runtime, metrics, optim 등)
│   ├── data/                       # 데이터셋 정의 (전처리, 경로, selector)
│   │   ├── mvtec.yaml              # MVTec AD 데이터셋 및 category selector
│   │   ├── btad.yaml               # (v0.3 확장)
│   │   └── visa.yaml               # (v0.3 확장)
│   ├── models/                     # 모델 아키텍처 정의
│   │   ├── stfpm.yaml              # STFPM 모델 및 backbone selector
│   │   └── efficientad.yaml        # EfficientAD 모델 및 model_size selector
│   └── batch/                      # 다중 조건 배치 매니페스트
│       └── mvtec_matrix.yaml       # MVTec 다중 카테고리/모델 실행 매트릭스
├── classification/                 # Classification Task
│   ├── _base.yaml
│   ├── data/ (cifar10.yaml)
│   ├── models/ (resnet18.yaml)
│   └── batch/ (cifar10_matrix.yaml)
├── detection/ & segmentation/      # Detection / Segmentation Task
├── splits/                         # 데이터셋 분할 파일 (train/valid/test JSON)
│   ├── mvtec_bottle.json
│   └── ...
├── assets.yaml                     # 로컬 자산 경로 레지스트리
└── local.example.yaml              # 로컬 머신별 경로 오버라이드 템플릿
```

---

### 3.2. `scripts/` (실행 진입점)

사용자가 터미널에서 직접 호출하는 단일/다중 조건 실행 스크립트 및 도구 모음이다.

```text
scripts/
├── train.py                # UC-001: 단일 조건 학습 진입점
├── evaluate.py             # UC-002: 단일 조건 평가 진입점
├── predict.py              # UC-003: 단일 조건 추론 및 시각화 진입점
├── run_batch.py            # UC-004: 다중 조건 일괄 실행 진입점 (배치 러너)
├── check_assets.py         # UC-005: 로컬 데이터셋/백본 가중치 유효성 점검 유틸리티
└── report.py               # UC-006: 실행 결과 지표 취합 및 리더보드 출력 유틸리티
```

---

### 3.3. `src/tasks/<task>/` 표준 패키지 구조

모든 Task는 컴포넌트별로 하위 폴더를 구성하며, 각 모델은 `models/<model_name>/` 하위의 독립된 `nn.Module` 패키지로 작성된다.

```text
src/tasks/<task_name>/
├── __init__.py                     # Task 패키지 진입점 (하위 모듈 자동 등록)
│
├── adapters/                       # Task 어댑터 패키지 (Lifecycle & Step 관리)
│   ├── __init__.py
│   ├── base.py                     # Task 공통 베이스 어댑터 (예: AnomalyAdapter)
│   └── <model_name>.py             # (선택) 모델별 전용 훅 어댑터 (stfpm.py, efficientad.py)
│
├── datasets/                       # 데이터셋 로더 패키지
│   ├── __init__.py
│   ├── base.py                     # 데이터셋 베이스 클래스
│   └── <dataset_name>.py           # 구체 데이터셋 클래스 (mvtec.py, btad.py 등)
│
├── models/                         # [SSOT] 모델 패키지 (모델별 개별 폴더)
│   ├── __init__.py                 # 하위 모델 팩토리 일괄 임포트
│   ├── <model_name_1>/             # 개별 모델 독립 패키지 (예: stfpm/)
│   │   ├── __init__.py             # @MODELS.register 팩토리 함수
│   │   ├── model.py                # class Model(nn.Module) - 순수 PyTorch 네트워크
│   │   ├── loss.py                 # 전용 Loss 모듈
│   │   └── <submodule>.py          # 서브모듈 (anomaly_map, components 등)
│   └── <model_name_2>/             # 개별 모델 독립 패키지 (예: efficientad/)
│       ├── __init__.py
│       ├── model.py                # class Model(nn.Module)
│       └── networks.py             # PDN, AutoEncoder 모듈
│
├── transforms/                     # 전처리 / 증강 파이프라인 패키지
│   ├── __init__.py
│   └── default.py                  # 표준 전처리 빌더 (@TRANSFORMS.register)
│
├── metrics/                        # 평가 지표 패키지 (torchmetrics 기반)
│   ├── __init__.py
│   ├── <metric_1>.py               # 개별 지표 클래스 (@METRICS.register)
│   └── <metric_2>.py
│
├── losses/                         # 공통/복합 손실 함수 패키지
│   ├── __init__.py
│   └── <loss_name>.py
│
└── postprocess/                    # 후처리 및 시각화 패키지
    ├── __init__.py
    ├── visualizer.py               # 결과 이미지 / 히트맵 오버레이 렌더러
    └── smoother.py                 # 가우시안 블러, 정규화, 마스크 생성기
```

---

### 3.4. `src/core/` (프레임워크 코어 엔진)

Task 종류나 모델명에 대한 분기문이 일절 없는 순수 task-agnostic 공통 엔진이다.

```text
src/core/
├── engine.py           # Trainer (pure-PyTorch fit, evaluate, predict 공통 루프)
├── config.py           # Config 로드, Selector 해석(apply_selectors), 병합 및 검증
├── checkpoint.py       # Checkpoint 저장/로드 및 RNG 상태 관리
├── registry.py         # Module 레지스트리 (MODELS, DATASETS, ADAPTERS, METRICS 등)
├── adapter.py          # Base TaskAdapter 및 Lifecycle Hook 규격 정의
├── context.py          # ExecutionContext (device, amp, seed, config 컨테이너)
├── builders.py         # build_model, build_optimizer, build_scheduler 팩토리
├── logger.py           # 콘솔 로거 및 metrics.json 기록기
├── paths.py            # 로컬 자산 경로 검증 헬퍼
└── errors.py           # ConfigError, AssetError 등 공통 예외 클래스
```

---

## 4. 계층 간 상호작용 및 데이터 흐름

```text
[scripts/ (CLI 진입점)]
       │
       ▼
[src/core/config.py] ◄─── [configs/<task>/ (data / model / batch YAML)]
  (Selector 해석 및 병합)
       │
       ▼
[src/core/engine.py (Trainer)]
       │
       ├───► [src/tasks/<task>/adapters/ (Lifecycle & Step Hook)]
       │            │
       │            ├───► [src/tasks/<task>/datasets/ (DataLoader)]
       │            ├───► [src/tasks/<task>/transforms/ (Preprocessing)]
       │            ├───► [src/tasks/<task>/metrics/ (TorchMetrics)]
       │            └───► [src/tasks/<task>/postprocess/ (Visualizer)]
       │
       └───► [src/tasks/<task>/models/<model>/ (SSOT Pure PyTorch nn.Module)]
```

1. **설정 합성**: `scripts/train.py`가 `--data`와 `--model` YAML을 로드하고 `apply_selectors`로 CLI 인자를 주입하여 완전한 Config를 생성한다.
2. **인스턴스화**: `src/core/registry.py`를 통해 `models/<model>/`의 `nn.Module` 인스턴스, 데이터로더, 어댑터를 생성한다.
3. **공통 루프 실행**: `src/core/engine.py::Trainer`가 task-agnostic하게 학습/평가 루프를 실행하며, 모델별 특수 lifecycle은 `TaskAdapter`의 hook을 통해 호출된다.
4. **산출물 저장**: `outputs/<run_name>/` 하위에 체크포인트(`checkpoints/best.pth`), 로그, 지표(`metrics_*.json`), 시각화 이미지가 저장된다.

---

작성일: 2026-08-22
문서 위치: `docs/guides/structure.md`
