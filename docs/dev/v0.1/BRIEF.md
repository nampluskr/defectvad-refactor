# BRIEF — Anomaly Detection Integration on `cv_boilerplate`

## 핵심 사용자 의도 / 목적

- 범용 Computer Vision 실행 프레임워크 `cv_boilerplate` 위에서 anomalib 기반 SOTA anomaly detection 모델을 학습·평가·추론·벤치마크한다.
- anomalib의 pure-PyTorch 모델 구현은 SSOT로서 수정 없이 그대로 사용한다.
- 실행 lifecycle은 `cv_boilerplate`가 담당하도록 통합한다.
- anomalib reference 수준의 성능 재현으로 통합의 정확성을 검증한다.

## 반드시 지켜야 할 세 가지 원칙

이 프로젝트의 대상 모델은 [anomalib `src/anomalib/models/image`](https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image)의 SOTA 모델이다. anomalib은 image와 video를 모두 다루지만 **이 프로젝트는 image로 한정한다**. 아래 세 원칙은 어떤 경우에도 예외 없이 지켜진다. 원칙 1과 2는 **비대칭**이다. 모델 코드는 불가침이고, boilerplate는 개선 대상이다.

### 원칙 1 — anomalib 모델 코드는 SSOT이며 수정하지 않는다

- anomaly detection 모델은 순수 PyTorch 코드여야 하며, anomalib에 구현된 코드를 **그대로 가져온다**.
- `torch_model.py`를 비롯한 모델 네트워크·알고리즘 코드는 **절대로 수정되어서는 안 된다**. `loss.py`, `anomaly_map.py` 등 함께 복사하는 구성요소도 동일하다.
- anomalib의 네트워크와 알고리즘이 SSOT이며, 어떤 이유로도 훼손되지 않아야 한다.
- 이 원칙은 사용자뿐 아니라 **무엇보다 Claude / Codex 등 AI 에이전트에게 우선 적용된다**. AI 에이전트가 원본 모델 코드를 수정하지 않는 것이 이 프로젝트의 가장 중요한 원칙이다.
- 리팩터링, 코드 스타일 통일, 타입 힌트 추가, 사소한 정리를 포함해 원본을 손대는 모든 행위를 금지한다. 통합 과정에서 문제가 발생하면 모델 코드가 아니라 boilerplate 쪽을 수정해 해결한다.

#### 유일한 예외 — 모듈 및 라이브러리 경로

개별 모델은 하위 components나 외부 라이브러리에 의존할 수 있다. `torch_model.py`에서 수정 가능한 부분은 **모듈 및 라이브러리 경로로 한정한다**.

- 허용: import 경로 조정 (예: `from anomalib.models.components import ...` → 이 프로젝트에 복사해 온 components 경로)
- 금지: 그 외 모든 변경 — 네트워크 구조, 연산, 하이퍼파라미터 기본값, 함수·클래스 시그니처, 반환 형식
- 모델이 의존하는 anomalib 하위 components 역시 동일한 원칙으로 그대로 복사하며, 경로 외에는 수정하지 않는다.
- components를 배치하는 경로는 anomalib의 디렉터리 구조와 동일할 필요가 없다. 이 프로젝트에 맞는 구조로 자유롭게 배치하고, import 경로만 그에 맞춰 조정한다.

### 원칙 2 — Lightning은 절대 사용하지 않는다

- Lightning 라이브러리는 **절대로 사용하지 않는다**. 의존성으로 추가하지 않고, import하지 않으며, 어떤 실행 경로에서도 호출하지 않는다.
- 모델은 오직 순수 PyTorch 기반 boilerplate 위에서 학습·평가·추론되어야 한다.
- `lightning_model.py`는 복사 대상이 아니라 **참고 대상**이다. 그 안의 optimizer, scheduler, post-processing, hooks 정의는 필요에 따라 boilerplate의 adapter와 config에 반영한다.
- Lightning으로 구현되어 있던 부분이 boilerplate에 올바르게 반영되어, **anomalib와 동일한 방식·동일한 성능**을 나타내야 한다.
- boilerplate는 고정된 것이 아니다. 성능 재현과 다양한 SOTA 모델의 평가를 위해 **얼마든지 수정·개선될 수 있다**.

### 원칙 3 — 데이터셋과 pretrained 가중치는 로컬 폴더의 것을 사용한다

- 데이터셋과 pretrained 가중치는 **로컬 폴더에 저장된 것을 사용해야 한다**. 실행 중 자동 다운로드에 의존하지 않는다.

| 자산 | 경로 |
|---|---|
| 데이터셋 | `/mnt/d/datasets` |
| pretrained 가중치 | `/mnt/d/backbones` |

- 예: MVTec AD는 `/mnt/d/datasets/mvtec`, EfficientAD의 auxiliary 데이터는 `/mnt/d/datasets/imagenette2`, EfficientAD pretrained teacher는 `/mnt/d/backbones/efficientad_pretrained_weights`, STFPM의 ResNet18 backbone은 `/mnt/d/backbones/resnet18-f37072fd.pth`를 사용한다.
- anomalib 원본 코드에 포함된 다운로드 로직은 모델 코드를 수정해 제거하지 않는다(원칙 1). 로컬 자산을 지정해 다운로드 경로를 타지 않도록 adapter 또는 boilerplate에서 처리한다(원칙 2).
- 경로는 config로 지정할 수 있어야 하며, 코드에 하드코딩하지 않는다.

#### 오프라인 환경 전제

이 프로젝트는 **인터넷이 연결되지 않은 로컬 환경에서 실행**되어야 한다. 따라서 로컬 자산 사용은 권장이 아니라 필수 조건이다.

- AI 에이전트는 필요한 데이터셋, 모델, 라이브러리 등을 **자동으로 다운로드하거나 설치해서는 안 된다**.
- 필요한 자산이 로컬에 없으면 임의로 받아오지 말고, **무엇이 어느 경로에 필요한지 사용자에게 알리고 CLI 환경에서 직접 다운로드·설치하도록 요청한다**.
- 자산이 준비되기 전까지는 해당 작업을 진행하지 않고 대기한다. 다운로드가 필요한 코드 경로를 우회하기 위해 모델 코드를 수정하는 것도 금지한다(원칙 1).

## 모델 추가 예시

새 모델 추가 요청("`xxx` 모델을 추가해 주세요")을 받은 에이전트는 아래 예시를 참고하여 anomalib에서 해당 모델 코드를 복사해 온다. 구체적인 경로와 adapter 설계는 [SPEC.md](SPEC.md)를 따른다.

### 공통 절차

1. anomalib `src/anomalib/models/image/<model>/`의 파일 구성을 확인한다. 모델마다 파일 구성이 다르다.
2. `lightning_model.py`를 **제외한** 순수 PyTorch 파일을 `upstream/<model>/`에 **수정 없이** 복사한다.
3. `lightning_model.py`는 복사하지 않고 **읽어서**, optimizer/scheduler는 config 값으로, post-processing과 hooks는 모델별 adapter로 옮겨 적는다.
4. upstream 출처(anomalib commit)와 라이선스를 기록한다.
5. 모델이 의존하는 하위 components를 확인해 함께 복사하고, import 경로만 이 프로젝트 구조에 맞게 조정한다.
6. 복사해 온 파일은 경로를 제외하고 이후 어떤 작업에서도 수정하지 않는다(원칙 1). 동작하지 않으면 adapter 또는 boilerplate를 고친다(원칙 2).

### 예시 1 — STFPM

anomalib `models/image/stfpm/`: `torch_model.py`, `loss.py`, `anomaly_map.py`, `lightning_model.py`, `__init__.py`, `README.md`

```text
upstream/stfpm/         # anomalib 원본 — 수정 금지
├── torch_model.py      # STFPMModel: teacher/student feature extractor
├── loss.py             # STFPMLoss
└── anomaly_map.py      # AnomalyMapGenerator

adapters/stfpm.py       # 이 프로젝트에서 작성 — lightning_model.py를 참고하여 재구성
```

`lightning_model.py`에서 옮겨야 할 내용:

- optimizer: SGD (lr 0.4, momentum 0.9, weight_decay 0.001)
- scheduler: 없음
- train step: teacher/student feature를 추출해 `STFPMLoss`로 계산
- eval step: anomaly map과 score 산출

### 예시 2 — EfficientAD

anomalib `models/image/efficient_ad/`: `torch_model.py`, `lightning_model.py`, `__init__.py`, `README.md`
(STFPM과 달리 `loss.py`/`anomaly_map.py`가 별도 파일로 분리되어 있지 않고 `torch_model.py`에 포함된다.)

```text
upstream/efficient_ad/  # anomalib 원본 — 수정 금지
└── torch_model.py      # teacher/student/autoencoder + loss + map 로직 포함

adapters/efficientad.py # 이 프로젝트에서 작성 — lightning_model.py를 참고하여 재구성
```

`lightning_model.py`에서 옮겨야 할 내용:

- optimizer: Adam (student + autoencoder 파라미터, lr 1e-4, weight_decay 1e-5)
- scheduler: StepLR — 전체 학습의 95% 시점에 lr 0.1배
- 학습 전 준비(`on_train_start` 상당): teacher pretrained weight 로드, ImageNette auxiliary 데이터 준비, teacher feature의 channel mean/std 계산
- 검증 전 준비(`on_validation_start` 상당): 정상 이미지로 feature map의 90% / 99.5% 분위수 계산 후 모델에 반영
- 제약: batch size 1, normalization transform 미사용

> EfficientAD는 STFPM에 없는 auxiliary 데이터 로딩과 학습 전후 통계 계산 단계를 요구한다. 이런 모델별 lifecycle 차이는 공통 engine이 아니라 각 모델의 adapter에서 흡수한다.