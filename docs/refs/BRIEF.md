# BRIEF — Anomaly Detection Integration on `cv_boilerplate`

## 1. 문서 목적

이 문서는 기존 Vision Anomaly Detection 레거시 프로젝트를 재정비하여, 범용 Computer Vision 실행 프레임워크인 `cv_boilerplate` 위에서 SOTA anomaly detection 모델을 학습·평가·추론·벤치마크하기 위한 **사용자 의도, 프로젝트 방향, 범위, 설계 원칙, 성공 기준**을 정의한다.

이 문서는 구현 설계서가 아니다. 세부 API, 클래스 구조, 파일 배치, Phase 순서 등은 후속 문서에서 현재 저장소의 실제 코드를 분석한 뒤 결정한다.

- 이 문서가 답하는 질문: **왜 만드는가 / 무엇을 달성하려는가 / 무엇을 유지하고 무엇을 버릴 것인가**
- 이 문서가 답하지 않는 질문: **구체적으로 어떤 클래스와 API로 어떻게 구현할 것인가**
- 우선순위: **이 문서에 명시된 사용자 의도 > 현재 `cv_boilerplate` 구조 > 레거시 구현 방식**
- 레거시 코드는 요구사항과 시행착오를 이해하기 위한 참고 자료이며, 그대로 보존하거나 리팩터링해야 하는 설계 기준은 아니다.

이후 작성되는 `PRD.md`, `SPEC.md`, `PLAN.md`, `backlog.json`은 모두 이 문서를 상위 의도와 범위의 source of truth로 사용한다.

---

## 2. 배경

사용자는 SOTA Vision Anomaly Detection 모델을 직접 학습하고 평가하기 위해 다음 레거시 프로젝트들을 작성해 왔다.

- `https://github.com/nampluskr/defectvad`
- `https://github.com/namplus-bit/defectvad_dev`
- `https://github.com/namplus-bit/anomaly_detection_dev`

이 프로젝트들은 AI coding agent를 사용하기 이전에 웹 기반 AI에게 질의하고, 제안된 코드를 복사·붙여넣고 수정하는 방식으로 발전했다.

그 과정에서 다음 목적은 일관되게 유지되었다.

1. anomalib에 구현된 SOTA anomaly detection 모델을 활용한다.
2. 모델의 핵심 알고리즘은 가능한 한 anomalib의 순수 PyTorch 구현을 그대로 사용한다.
3. anomalib의 Lightning 기반 학습·평가·추론 실행 계층은 사용하지 않는다.
4. 학습·평가·추론 과정을 직접 이해하고 수정 가능한 pure-PyTorch 실행 흐름으로 구성한다.
5. MVTec AD 등 표준 benchmark를 통해 원본 구현 수준의 성능을 확인한다.

그러나 레거시 프로젝트들은 anomaly detection 내부에 자체 `BaseTrainer`, `Evaluator`, `Factory`, Dataset/Config 구조 등을 반복적으로 만들면서, 모델 구현과 범용 실행 프레임워크의 책임이 혼재되었다.

이후 `cv_boilerplate` 프로젝트에서는 이러한 학습·평가·추론·벤치마크 구조를 anomaly detection에 한정하지 않고 Classification, Segmentation, Detection 등 다른 Computer Vision task에도 적용 가능한 범용 구조로 일반화하는 방향이 확립되었다.

따라서 이번 작업의 본질은 **기존 `defectvad` 프레임워크를 더 잘 리팩터링하는 것**이 아니다.

이번 작업은:

> **anomalib의 SOTA anomaly detection 순수 PyTorch 모델 구현을 가능한 한 보존하면서, 실행 lifecycle은 범용 `cv_boilerplate`가 담당하도록 통합하고, anomalib reference 수준의 성능 재현으로 그 통합의 정확성을 검증하는 것**

이다.

---

## 3. 핵심 사용자 의도

사용자는 anomaly detection 모델이나 dataset의 내부 구조를 매번 이해해야만 실험할 수 있는 프로젝트를 원하지 않는다.

사용자가 원하는 최종 경험은 다음과 같다.

1. 사용할 task, dataset, model, 주요 학습 조건을 설정한다.
2. 공통된 방식으로 `train`, `evaluate`, `predict`, `benchmark`를 실행한다.
3. 개별 모델의 내부 architecture, loss 계산 방식, anomaly map 생성 방식, post-processing, 모델별 학습 lifecycle 차이는 integration 계층이 처리한다.
4. 결과를 표준 metric과 benchmark 형태로 확인한다.
5. 동일 조건의 anomalib reference 결과와 비교하여 구현 정확성을 판단한다.

즉 사용자-facing workflow는 가능한 한 일관되어야 하며, 모델마다 서로 다른 내부 구현과 lifecycle은 사용자에게 노출되는 필수 지식이 되어서는 안 된다.

핵심 원칙은 다음 문장으로 요약한다.

> **Uniform user workflow, heterogeneous model internals.**

---

## 4. 프로젝트 목적

### 4.1 1차 목적

`cv_boilerplate`를 기반으로 Vision Anomaly Detection 모델의 다음 lifecycle을 수행한다.

- Training
- Evaluation
- Inference / Prediction
- Benchmark

### 4.2 모델 통합 목적

Anomaly Detection 모델의 핵심 알고리즘은 anomalib의 순수 PyTorch 구현을 최대한 그대로 사용한다.

대표적으로 다음과 같은 구현 요소가 이에 해당할 수 있다.

- `torch_model.py`
- `loss.py`
- `anomaly_map.py`
- feature extractor
- memory bank 관련 로직
- 모델 고유 component
- 모델 고유 preprocessing / post-processing 로직 중 알고리즘 의미에 직접 관련된 부분

가능하면 anomalib upstream과의 차이를 작게 유지하여, 모델 알고리즘 자체를 다시 구현하거나 재해석하면서 발생하는 오류를 줄인다.

### 4.3 실행 프레임워크 목적

다음 계층은 anomalib의 실행 프레임워크를 사용하지 않고 `cv_boilerplate`가 담당한다.

- Training lifecycle
- Validation / Evaluation lifecycle
- Prediction lifecycle
- Benchmark orchestration
- Device handling
- Checkpoint handling
- Config orchestration
- Dataset/DataLoader lifecycle
- Metric orchestration
- Result/output management
- 재현성 관리

즉 anomalib에서는 **무엇을 계산하는가(model algorithm)** 를 가져오고, `cv_boilerplate`는 **어떻게 실행하는가(execution lifecycle)** 를 소유한다.

---

## 5. anomalib과의 관계

### 5.1 재사용할 것

anomalib는 SOTA Vision Anomaly Detection 알고리즘의 reference implementation source로 사용한다.

원칙:

- 순수 PyTorch 모델 구현은 가능한 한 그대로 복사·사용한다.
- 알고리즘 의미를 변경하는 리팩터링은 최소화한다.
- `cv_boilerplate`와 연결하기 위한 adaptation은 모델 코드 자체를 대규모로 바꾸기보다 adapter/hook/integration 계층에서 해결하는 것을 우선한다.
- 필요한 경우 anomalib의 source code와 공식 benchmark/config를 reference로 사용한다.

### 5.2 사용하지 않을 것

다음 anomalib 실행 계층은 이번 프로젝트의 runtime으로 사용하지 않는다.

- Lightning 기반 model wrapper
- Lightning `Trainer`
- anomalib `Engine`
- Lightning callback lifecycle
- anomalib CLI orchestration
- 기타 학습·평가·추론 control flow를 anomalib에 위임하는 구조

목표는 anomalib 전체를 제거하는 것이 아니라, **알고리즘 구현과 실행 프레임워크의 경계를 명확히 하는 것**이다.

### 5.3 목표 표현

이 프로젝트를 다음과 같이 표현하지 않는다.

> anomalib을 pure PyTorch로 다시 구현한다.

보다 정확한 표현은 다음과 같다.

> anomalib의 pure-PyTorch anomaly detection 모델 구현을 `cv_boilerplate` runtime에 최소 수정으로 통합한다.

---

## 6. `cv_boilerplate`와의 관계

`cv_boilerplate`는 anomaly detection 전용 framework가 아니라 여러 Computer Vision task를 위한 범용 실행 기반이다.

Anomaly Detection은 그 위에서 동작하는 하나의 task/domain integration으로 취급한다.

개념적 책임 경계는 다음과 같다.

```text
cv_boilerplate
│
├── config / experiment orchestration
├── train lifecycle
├── evaluate lifecycle
├── predict lifecycle
├── benchmark lifecycle
├── device / checkpoint / reproducibility
├── common dataset/model execution contracts
└── task integration boundary
        │
        └── Anomaly Detection
              ├── dataset semantics
              ├── anomaly metrics
              ├── anomaly post-processing
              ├── model-specific adaptation
              └── anomalib pure-PyTorch models
```

### 6.1 기본 방향

- 공통 engine은 가능한 한 task-agnostic이어야 한다.
- anomaly detection 때문에 공통 engine 내부에 모델명 또는 task명 기반 분기를 확산시키지 않는다.
- 모델마다 다른 lifecycle은 adapter, hook, strategy 또는 이에 상응하는 extension point로 흡수한다.
- 정확한 extension API는 이 BRIEF에서 고정하지 않고 `SPEC.md`에서 실제 `cv_boilerplate` 코드를 분석한 뒤 결정한다.

### 6.2 중요한 검증 의미

Anomaly Detection은 `cv_boilerplate`의 범용성을 검증하는 중요한 stress test이기도 하다.

Anomaly Detection 모델은 일반적인 supervised gradient-training lifecycle과 다른 특성을 포함할 수 있다.

예:

- teacher/student 구조
- auxiliary autoencoder
- memory bank 구축
- feature statistics fitting
- 사실상 gradient training이 없는 모델
- training과 inference에서 다른 forward/output 의미
- dataset 기반 normalization/statistics 계산
- image-level score와 pixel-level anomaly map 동시 산출
- threshold 또는 post-processing calibration

따라서 여러 anomalib 모델을 하나의 사용자 workflow 아래에서 처리할 수 있다면, 이는 `cv_boilerplate`의 abstraction이 단순 classification training template을 넘어 범용 CV lifecycle로 작동하는지 검증하는 근거가 된다.

---

## 7. 목표 사용자 경험

사용자는 실험을 수행하기 위해 다음과 같은 모델별 내부 세부사항을 반드시 이해할 필요가 없어야 한다.

- 모델 내부 architecture
- teacher/student/autoencoder 구성 방식
- 모델-specific loss invocation
- model output parsing 방식
- anomaly map 생성 방식
- model-specific optimizer/scheduler 요구사항
- memory bank/statistics 구축 절차
- dataset 내부 디렉터리 parsing 규칙
- ground-truth mask representation 변환
- metric 입력 형식 변환
- 모델별 validation/post-processing lifecycle

사용자는 가능한 한 다음 수준의 개념만 선택하면 된다.

```yaml
task: anomaly_detection

dataset:
  name: mvtec
  category: bottle

model:
  name: stfpm

trainer:
  epochs: 100

device: cuda
```

위 예시는 사용자 경험의 방향을 보여주기 위한 개념 예시이며, 실제 config schema나 CLI 문법을 이 BRIEF에서 확정하지 않는다.

---

## 8. Benchmark의 역할

Benchmark는 단순한 편의 기능이나 leaderboard 생성 기능이 아니다.

이번 프로젝트에서 benchmark는 **integration correctness를 확인하는 핵심 검증 수단**이다.

### 8.1 Reference equivalence

동일하거나 합리적으로 동등한 조건에서 다음을 비교한다.

```text
anomalib reference implementation
              │
              │ same model family
              │ same dataset/category
              │ equivalent preprocessing
              │ equivalent hyperparameters
              │ equivalent metric definition
              ▼
cv_boilerplate + anomalib PyTorch model integration
```

비교 대상에는 모델 특성에 따라 다음 metric이 포함될 수 있다.

- Image-level AUROC
- Pixel-level AUROC
- AUPRO / PRO 계열 metric
- F1 / threshold 기반 metric
- 기타 해당 anomalib 모델 또는 benchmark에서 사용하는 표준 metric

정확한 metric 세트와 허용 오차(tolerance)는 모델과 anomalib reference 환경을 분석하여 `PRD.md` 또는 `SPEC.md`에서 명시한다.

### 8.2 성공의 의미

다음은 충분한 성공 기준이 아니다.

> 코드가 실행된다.

목표는 다음에 가깝다.

> `cv_boilerplate` 기반 구현이 anomalib reference implementation과 비교 가능한 수준의 benchmark 성능을 재현한다.

성능 차이가 발생하면 단순히 모델의 차이로 간주하지 않고 다음을 검토한다.

- preprocessing 차이
- augmentation 차이
- optimizer / scheduler 차이
- epoch / batch 조건 차이
- pretrained weight 차이
- anomaly score normalization 차이
- post-processing 차이
- threshold 계산 차이
- metric 구현 차이
- evaluation protocol 차이

즉 성능 재현은 model code뿐 아니라 전체 experiment protocol의 동등성을 검증한다.

---

## 9. 프로젝트 목표

### G-01. 범용 boilerplate 기반 실행

Anomaly Detection이 `cv_boilerplate`의 공통 workflow 안에서 학습·평가·추론·벤치마크될 수 있어야 한다.

### G-02. anomalib 알고리즘 보존

anomalib의 검증된 pure-PyTorch 모델 알고리즘은 가능한 한 수정 없이 재사용한다.

### G-03. Lightning 실행 계층 제거

anomalib의 Lightning/Engine 기반 학습·평가·추론 orchestration에는 의존하지 않는다.

### G-04. 사용자-facing interface 일관성

사용자는 모델마다 다른 내부 training/evaluation implementation을 직접 다루지 않고 공통된 실행 방식으로 실험할 수 있어야 한다.

### G-05. 모델별 lifecycle 수용

일반 gradient training뿐 아니라 memory bank, feature statistics, teacher/student 등 서로 다른 anomaly detection lifecycle을 수용할 수 있어야 한다.

### G-06. Reference 성능 재현

표준 benchmark에서 anomalib reference와 비교 가능한 수준의 성능을 재현해야 한다.

### G-07. 모델 추가 비용 최소화

새 anomalib 모델을 추가할 때 공통 framework를 복제하거나 재작성하지 않고, 모델 코드 복사와 최소한의 integration 작업으로 추가할 수 있어야 한다.

### G-08. Dataset 독립성

모델 코드와 metric/evaluation 로직은 특정 dataset의 디렉터리 구조에 강하게 결합되지 않아야 한다.

### G-09. 재현 가능한 benchmark

동일 config, seed, dataset, weights 및 protocol로 실험 결과를 반복 검증할 수 있어야 한다.

### G-10. 제한된 네트워크 환경 고려

필요한 dataset과 pretrained weights를 로컬에 준비한 경우 인터넷 접근 없이 학습·평가·추론·벤치마크를 수행할 수 있는 방향을 유지한다.

---

## 10. 비목표 / Non-Goals

이번 프로젝트의 목적은 다음이 아니다.

### NG-01. anomalib 전체 재구현

anomalib의 Engine, CLI, callback, deployment 기능 전체를 복제하지 않는다.

### NG-02. 새로운 anomaly detection 알고리즘 연구

새로운 SOTA algorithm 자체를 설계하는 것이 이번 refactoring/integration의 주목적은 아니다.

### NG-03. anomalib 모델의 대규모 재작성

`torch_model.py`, `loss.py`, `anomaly_map.py` 등을 프로젝트 스타일에 맞게 전면 rewrite하는 것을 목표로 하지 않는다.

### NG-04. `defectvad` 내부 framework 개선 자체

레거시 `BaseTrainer`, `Evaluator`, `Factory` 등을 더 정교한 anomaly-specific framework로 발전시키는 것이 최종 목적이 아니다.

범용 lifecycle 책임은 가능한 한 `cv_boilerplate`로 이동한다.

### NG-05. Anomalib API compatibility

anomalib의 Lightning API 또는 Engine API와 동일한 public interface를 제공하는 것은 요구사항이 아니다.

### NG-06. 범용 딥러닝 프레임워크 개발

PyTorch 위에 새로운 범용 딥러닝 프레임워크를 만드는 것이 목적이 아니다. 필요한 abstraction은 `cv_boilerplate`의 기존 철학과 범위 내에서 최소화한다.

### NG-07. 모든 모델을 하나의 고정된 `training_step`에 강제

모델의 lifecycle 차이를 무시하고 모든 anomaly detection 알고리즘을 동일한 gradient loop에 억지로 맞추지 않는다.

### NG-08. 불필요한 enterprise/MLOps 기능

분산 실험 플랫폼, 모델 registry 서비스, cloud orchestration 등은 명시적인 별도 요구가 없는 한 이번 범위에 포함하지 않는다.

---

## 11. 설계 원칙

### P-01. Boilerplate owns the lifecycle

학습·평가·추론·벤치마크의 실행 lifecycle은 `cv_boilerplate`가 소유한다.

### P-02. Task integration owns domain semantics

Anomaly Detection 특유의 dataset semantics, metric, post-processing 및 모델별 차이는 task/model integration 계층이 담당한다.

### P-03. Preserve upstream algorithms

anomalib의 순수 PyTorch 알고리즘 코드는 가능한 한 그대로 유지한다.

### P-04. Minimal adaptation

framework에 맞추기 위해 알고리즘 본체를 변경하기보다 최소한의 adapter/hook/integration 코드를 우선한다.

### P-05. Reference performance is a contract

통합 성공 여부는 단순 실행 성공보다 anomalib reference 성능 재현을 중요하게 판단한다.

### P-06. Engine remains task-agnostic

공통 engine에 anomaly detection 또는 특정 모델 이름 기반의 조건 분기를 반복적으로 추가하지 않는다.

### P-07. Explicit execution over hidden magic

사용자가 모든 모델 내부를 알 필요는 없지만, framework의 training/evaluation flow 자체는 추적 가능하고 테스트 가능해야 한다.

### P-08. Avoid speculative abstractions

실제 두 개 이상의 모델/상황에서 필요성이 확인되지 않은 abstraction을 미리 만들지 않는다.

### P-09. Legacy is evidence, not architecture

레거시 프로젝트는 요구사항, 시행착오, reference 코드의 출처로 사용하되 기존 클래스/폴더 구조를 유지해야 할 이유로 사용하지 않는다.

### P-10. Verify before generalize

먼저 대표 모델에서 reference equivalence를 검증하고, 그 결과를 바탕으로 공통 abstraction을 확장한다.

---

## 12. 레거시 프로젝트의 역할

### 12.1 `defectvad`

주요 참고 가치:

- pure-PyTorch `BaseTrainer`/`Evaluator`를 만들려 했던 목적
- MVTec / VisA / BTAD dataset 처리 경험
- anomalib 모델의 `torch_model.py`, `loss.py`, `anomaly_map.py` 복사·사용 방식
- STFPM, EfficientAD 등 모델-specific trainer 작성 경험
- 로컬 backbone/dataset 경로 및 offline 실행 경험
- train/evaluate/predict 분리 경험

사용 방법:

- 동작했던 코드와 알고리즘 연결 방식을 참고한다.
- 범용 framework 구조를 그대로 유지할 필요는 없다.

### 12.2 `defectvad_dev`

주요 참고 가치:

- anomalib의 Lightning 계층을 pure PyTorch로 대체하려는 초기 의도
- 모델 구현과 trainer 구현을 구분하려 한 설계 시도
- 실제 모델 학습을 맞추기 위해 필요했던 모델-specific lifecycle 정보

### 12.3 `anomaly_detection_dev`

주요 참고 가치:

- SOTA Anomaly Detection 실험 프레임워크를 반복 개선하려 한 흔적
- 테스트/검증 중심 개발 방식에 대한 시도
- 이전 프로젝트에서 해결되지 않았던 구조적 문제를 찾기 위한 참고 자료

### 12.4 `cv_boilerplate`

이번 작업의 실행 기반이 되는 현재 프로젝트.

주요 역할:

- task-agnostic training/evaluation/prediction/benchmark lifecycle
- 공통 config와 experiment flow
- 여러 CV task가 공유하는 실행 infrastructure
- task/model integration extension point 제공

이번 작업에서는 `cv_boilerplate`의 현재 실제 구현을 분석한 뒤 필요한 확장 또는 수정 범위를 결정한다.

---

## 13. 초기 지원 범위에 대한 방향

정확한 모델/데이터셋 목록은 후속 요구사항 문서에서 확정한다.

다만 초기 통합은 서로 lifecycle 특성이 다른 대표 anomaly detection 모델을 포함해야 한다.

예를 들어 다음과 같은 축이 유용하다.

1. 일반적인 gradient training 모델
2. teacher/student 기반 모델
3. memory bank 또는 feature statistics 기반 모델
4. auxiliary dataset 또는 별도 calibration이 필요한 모델

기존 코드에서 이미 다룬 STFPM과 EfficientAD는 우선 검토 가치가 높지만, 이 BRIEF 자체가 특정 모델 집합을 최종 확정하지는 않는다.

Dataset 역시 MVTec AD를 첫 reference benchmark로 우선 검토하되, VisA/BTAD 확장 여부와 범위는 PRD에서 결정한다.

---

## 14. 성공 기준

프로젝트는 최소한 다음 상태를 지향한다.

1. Anomaly Detection 모델이 `cv_boilerplate`의 공통 실행 흐름에서 동작한다.
2. 사용자는 개별 모델 내부 구현을 직접 다루지 않고 train/evaluate/predict/benchmark를 실행할 수 있다.
3. anomalib의 핵심 pure-PyTorch 모델 구현이 대규모 rewrite 없이 재사용된다.
4. anomalib Lightning/Engine에 런타임 의존하지 않는다.
5. 서로 다른 anomaly detection lifecycle을 공통 framework의 extension mechanism으로 처리할 수 있다.
6. 표준 benchmark에서 anomalib reference와 비교 가능한 성능을 재현한다.
7. reference와 성능 차이가 발생한 경우 protocol 차이를 추적할 수 있는 config/result 기록이 남는다.
8. 새 anomalib 모델 추가 시 공통 engine을 복제하지 않는다.
9. 로컬 dataset과 pretrained weights가 준비된 환경에서 network download 없이 실험 가능하다.
10. 구현의 정확성을 unit/integration/benchmark 수준에서 단계적으로 검증할 수 있다.

정량적 metric tolerance, 실행 모델 수, benchmark dataset/category 범위 등은 PRD/SPEC에서 검증 가능한 acceptance criteria로 구체화한다.

---

## 15. 제약사항 및 고려사항

### 15.1 Pure PyTorch 실행 흐름

학습·평가·추론의 control flow는 PyTorch 수준에서 직접 관리한다. Lightning 등 상위 training framework를 새 runtime으로 도입하지 않는다.

### 15.2 Local asset 우선

Dataset과 pretrained weight는 프로젝트가 자동 다운로드하는 것을 기본 가정으로 하지 않는다. 로컬 자산을 명시적으로 사용할 수 있어야 한다.

### 15.3 Dependency 최소화

새로운 dependency는 실제 필요성이 있을 때 도입한다. anomalib 전체 runtime에 의존하는 방식으로 integration 문제를 우회하지 않는다.

### 15.4 Reference version 고정 필요

성능 재현성을 위해 benchmark 시 사용한 anomalib 버전/commit, dataset version, pretrained weights, preprocessing/config를 기록할 수 있어야 한다.

### 15.5 모델별 lifecycle 차이

모든 모델이 표준 epoch/optimizer loop를 따를 것이라고 가정하지 않는다.

### 15.6 성능 동등성의 현실적 정의

GPU, library version, numerical nondeterminism 등에 따라 bitwise 동일 결과를 요구하지 않는다. 대신 metric 기준의 합리적인 tolerance를 정의한다.

이 tolerance는 임의로 정하지 않고 anomalib reference run과 반복 실험 결과를 바탕으로 후속 문서에서 확정한다.

---

## 16. 의도적으로 아직 결정하지 않는 항목

다음은 현재 BRIEF에서 확정하지 않는다.

- `cv_boilerplate`의 정확한 anomaly adapter API
- Base class / protocol / hook의 구체적인 이름과 hierarchy
- 모델 registry 방식
- dataset registry 방식
- 실제 config schema
- CLI 명령 형식
- checkpoint 파일 구조
- output directory 구조
- metric library의 구체적 선택
- anomaly threshold 계산 정책
- normalization/post-processing 공통화 수준
- 최초 지원 anomalib 모델의 최종 목록
- 최초 benchmark category의 최종 목록
- anomalib reference version/commit
- reference metric tolerance
- 모델별 upstream source 파일을 repository에 vendor할지 별도 sync 구조를 둘지 여부
- anomalib 라이선스/attribution을 repository에서 관리하는 구체적인 방법
- `cv_boilerplate` 공통 코드 변경이 필요한 범위

이 항목들은 실제 repository 분석과 reference implementation 검토 후 `PRD.md` 및 `SPEC.md`에서 결정한다.

---

## 17. 후속 문서 체계

본 프로젝트는 다음 문서 chain을 사용한다.

```text
BRIEF.md
   │
   ▼
PRD.md
   │
   ▼
SPEC.md
   │
   ▼
PLAN.md
   │
   ▼
backlog.json
   │
   ▼
Implementation / Verification
```

각 문서는 이전 문서의 내용을 단순 반복하는 것이 아니라, 상위 문서의 의도를 다음 수준의 구체성으로 변환한다.

### 17.1 `BRIEF.md` — Why / Direction / Scope

현재 문서.

역할:

- 프로젝트가 존재하는 이유
- 사용자 의도
- 레거시 프로젝트의 의미
- `cv_boilerplate`와 anomalib의 역할 경계
- 목표와 비목표
- 설계 원칙
- 성공의 의미
- 제약사항
- 후속 단계로 미룬 결정

이 문서는 **사용자 의도와 프로젝트 방향의 source of truth**이다.

구현이 어렵다는 이유만으로 후속 문서에서 BRIEF의 핵심 목적을 임의 변경하지 않는다.

### 17.2 `PRD.md` — Product Requirements

BRIEF의 의도를 **검증 가능한 요구사항**으로 변환한다.

권장 구성:

| 구분 | 내용 |
|---|---|
| Functional Requirements | train/evaluate/predict/benchmark, model/dataset integration 등 반드시 제공해야 할 기능 |
| Non-Functional Requirements | 재현성, 확장성, 유지보수성, offline 실행, testability 등 품질 요구 |
| Constraints | pure PyTorch, anomalib Lightning 미사용, local asset 정책 등 강제 제약 |
| Out of Scope / Non-Goals | 이번 버전에서 구현하지 않을 기능 |
| Gap / Implementation Status | 현재 코드에 이미 있는 기능과 부족한 기능의 분석 결과 |
| Acceptance Criteria | 각 requirement가 완료되었다고 판단할 검증 조건 |

요구사항에는 추적 가능한 ID를 부여한다.

예:

- `FR-001`
- `FR-002`
- `NFR-001`
- `CON-001`

PRD는 가능한 한 **What**을 정의하며 구체적인 클래스/API 설계는 SPEC에 위임한다.

### 17.3 `SPEC.md` — Technical Specification

PRD requirement를 현재 repository 구조에 맞게 기술적으로 구체화한다.

작성 전에 최소한 다음을 분석해야 한다.

- 현재 `cv_boilerplate` codebase
- 현재 `cv_boilerplate`의 task contract / engine lifecycle
- anomaly detection 레거시 프로젝트
- 통합 대상 anomalib 버전과 해당 모델 source
- reference training/evaluation configuration

SPEC이 정의해야 할 내용의 예:

- component responsibilities
- interface / protocol / hook
- data flow
- model lifecycle
- dataset/target contract
- train/evaluate/predict 흐름
- checkpoint/state 관리
- metric 및 post-processing 흐름
- benchmark reference protocol
- error handling
- test interface
- dependency 관계
- PRD requirement와 SPEC section의 traceability

SPEC은 상상한 architecture를 먼저 정하는 문서가 아니다.

**현재 코드와 reference implementation을 분석한 후 가장 작은 변경으로 요구사항을 만족하는 기술 설계**를 작성한다.

### 17.4 `PLAN.md` — Implementation Plan

PRD와 SPEC을 실제 구현 순서로 변환한다.

PLAN은 dependency를 고려한 Phase로 구성한다.

각 Phase는 최소한 다음을 포함한다.

- Phase 목적
- 포함 범위
- 상세 구현 내용
- 변경 예상 영역
- 선행 조건 / dependency
- 완료 조건
- 검증 방법
- regression 범위
- 다음 Phase로 넘어가기 위한 gate

Phase는 단순 기능 목록이 아니라 **독립적으로 검증 가능한 increment**가 되어야 한다.

대표적인 구조 예시는 다음과 같을 수 있으나, 실제 Phase는 SPEC 이후 결정한다.

```text
P1: Current architecture / reference baseline 확정
P2: Anomaly task common contract
P3: First reference model integration
P4: Different-lifecycle model integration
P5: Benchmark/reference equivalence
P6: Additional models/datasets and regression
```

이 예시는 방향을 설명하기 위한 것이며 현재 BRIEF에서 PLAN을 확정하지 않는다.

### 17.5 `backlog.json` — Execution and Progress State

`PLAN.md`의 Phase와 작업을 agent가 실행·추적할 수 있는 machine-readable work item으로 변환한다.

목적:

- 작업 단위 관리
- dependency 관리
- 진행 상태 관리
- verification 결과 관리
- 중단 후 재개
- agent 간 handoff
- 완료 조건 추적

각 item에는 필요에 따라 다음 정보가 포함될 수 있다.

```json
{
  "id": "P2-T03",
  "phase": "P2",
  "title": "Implement anomaly dataset adapter",
  "status": "pending",
  "depends_on": ["P2-T01"],
  "prd_refs": ["FR-003", "NFR-002"],
  "spec_refs": ["SPEC-4.2"],
  "verification": [
    "unit tests pass",
    "MVTec bottle smoke test passes"
  ]
}
```

정확한 JSON schema는 `PLAN.md` 작성 이후 결정한다.

`backlog.json`은 요구사항이나 설계의 source of truth가 아니다. **PLAN의 실행 상태 표현**이다.

---

## 18. 문서 간 우선순위와 변경 원칙

문서 간 의미 충돌이 있을 경우 다음 우선순위를 사용한다.

```text
사용자가 명시적으로 수정한 의도
        >
BRIEF.md
        >
PRD.md
        >
SPEC.md
        >
PLAN.md
        >
backlog.json
        >
현재 구현
        >
레거시 구현
```

단, 실제 코드 분석 결과 BRIEF의 가정이 사실과 다르거나 목표 달성이 불가능한 제약이 발견되면 임의로 하위 문서에서 우회하지 않는다.

다음 순서를 따른다.

1. 사실/제약을 명시한다.
2. BRIEF 의도에 미치는 영향을 설명한다.
3. 가능한 대안을 제시한다.
4. 사용자 결정이 필요한 경우 상위 문서부터 수정한다.
5. 수정된 상위 문서를 기준으로 하위 문서를 다시 동기화한다.

---

## 19. Codex 후속 작업 원칙

Codex가 이 BRIEF 이후 문서를 작성할 때 다음 원칙을 따른다.

### 19.1 먼저 읽을 자료

최소한 다음을 함께 검토한다.

1. 현재 프로젝트의 `BRIEF.md`
2. `cv_boilerplate`의 현재 source와 개발 문서
3. `defectvad`
4. `defectvad_dev`
5. `anomaly_detection_dev`
6. 통합 대상 anomalib의 공식 source/config/reference

### 19.2 레거시 해석 원칙

레거시 코드에 존재한다는 이유만으로 요구사항으로 승격하지 않는다.

반드시 다음 질문을 한다.

- 이것은 현재 사용자 의도에 필요한가?
- `cv_boilerplate`가 이미 담당하고 있는가?
- anomaly-specific responsibility인가?
- 과거 copy/paste 과정에서 생긴 우연한 구조인가?
- anomalib reference 성능 재현에 실제로 필요한가?

### 19.3 구조 추가 원칙

새 abstraction을 추가하기 전에 다음을 확인한다.

1. 기존 `cv_boilerplate` extension point로 해결 가능한가?
2. 최소 adapter로 해결 가능한가?
3. 두 개 이상의 모델에서 실제로 필요한 공통점인가?
4. 공통 engine을 task-specific하게 오염시키지 않는가?
5. reference 성능 검증을 어렵게 만들지 않는가?

---

## 20. 핵심 결정 요약

| 항목 | 결정 |
|---|---|
| 최상위 목표 | `cv_boilerplate` 위에서 SOTA Anomaly Detection 모델을 일관되게 train/evaluate/predict/benchmark |
| 모델 알고리즘 source | anomalib pure-PyTorch 구현 |
| 모델 코드 변경 원칙 | 가능한 한 최소 수정 / copy-and-adapt |
| 실행 lifecycle | `cv_boilerplate` |
| anomalib Lightning/Engine | 사용하지 않음 |
| 사용자 요구 지식 | 개별 model/dataset 내부 구조를 필수로 알 필요 없음 |
| framework 성격 | anomaly-specific framework가 아니라 범용 boilerplate의 task integration |
| correctness 기준 | anomalib reference와 비교 가능한 benchmark 성능 |
| 레거시 역할 | 요구사항/시행착오/reference 참고, architecture source of truth 아님 |
| 신규 모델 확장 | 공통 engine 재작성 없이 최소 integration으로 추가 |
| network 정책 | local dataset/weights 기반 offline 실행 가능 방향 |
| 현재 문서 역할 | 사용자 의도/방향/범위의 source of truth |
| 다음 문서 | `PRD.md` → `SPEC.md` → `PLAN.md` → `backlog.json` |

---

## 21. 한 문장 정의

> **Anomalib의 SOTA anomaly detection 순수 PyTorch 모델 구현은 가능한 한 그대로 유지하고, Lightning 기반 실행 계층은 범용 `cv_boilerplate`의 학습·평가·추론·벤치마크 lifecycle로 대체하며, 사용자가 모델·데이터셋 내부 구조를 몰라도 일관된 방식으로 실험할 수 있게 하고, anomalib reference 수준의 성능 재현으로 통합의 정확성을 검증한다.**

---

작성일: 2026-08-19  
문서 상태: Initial Brief  
후속 문서: `PRD.md` → `SPEC.md` → `PLAN.md` → `backlog.json`
