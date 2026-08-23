# Anomaly Detection 모델 가이드

이 문서는 현재 프로젝트에 통합된 anomalib 기반 anomaly detection 모델을 패러다임별로 분류하고 구현 현황과 실행 요구사항을 정리한다. 모델의 정량적 성능과 데이터셋별 평가는 다루지 않는다.

## 1. 전체 구현 현황

`구현됨`은 모델 패키지, registry factory, 전용 `TaskAdapter`, model config가 저장소에 통합되었다는 의미다. 성능 재현이나 사용자 데이터셋에 대한 품질 검증을 의미하지 않는다.

| 패러다임 | 모델 | 핵심 방식 | 구현 상태 | 학습 | 평가 | 추론 | 기본 backbone 또는 크기 | 추가 로컬 자산 |
|---|---|---|---|---|---|---|---|---|
| Teacher–Student / Knowledge Distillation | STFPM | 다중 계층 feature distillation | 구현됨 | 지원 | 지원 | 지원 | `resnet18` | backbone 가중치 |
| Teacher–Student / Reconstruction | EfficientAD | PDN teacher–student와 autoencoder | 구현됨 | 지원 | 지원 | 지원 | `small` | teacher 가중치, Imagenette |
| Normalizing Flow | FastFlow | feature map의 2D normalizing flow | 구현됨 | 지원 | 지원 | 지원 | `resnet18` | backbone 가중치 |
| Feature Embedding / Memory Bank | PatchCore | patch embedding memory bank와 최근접 이웃 탐색 | 구현됨 | 지원 | 지원 | 지원 | `wide_resnet50_2` | backbone 가중치 |

네 모델은 `src/tasks/anomaly/models/` 아래의 pure-PyTorch 모델과 `src/tasks/anomaly/adapters/` 아래의 lifecycle adapter로 구성된다. 학습 열의 `지원`은 gradient 학습만을 뜻하지 않는다. PatchCore의 학습 단계는 파라미터 최적화 대신 정상 이미지의 embedding을 수집하고 memory bank를 구축한다.

## 2. 모델 분류 체계

모델은 anomaly score를 만드는 주된 방식을 기준으로 하나의 대표 패러다임에 배치한다. 여러 방식을 결합한 모델은 보조 패러다임을 함께 표기한다.

### 2.1 Teacher–Student / Knowledge Distillation

고정된 teacher가 추출한 정상 feature를 student가 재현하도록 학습하고 두 출력의 차이를 anomaly signal로 사용한다.

- STFPM
- EfficientAD

EfficientAD는 teacher–student 구조와 autoencoder 기반 reconstruction을 결합하므로 `Reconstruction`을 보조 패러다임으로 함께 표기한다.

### 2.2 Normalizing Flow

사전 학습된 backbone feature를 invertible flow로 변환하여 정상 feature의 분포를 학습한다.

- FastFlow

### 2.3 Feature Embedding / Memory Bank

정상 이미지의 patch embedding을 memory bank에 저장하고 입력 patch와 가까운 정상 embedding 사이의 거리를 anomaly signal로 사용한다.

- PatchCore

## 3. 모델별 구현 개요

### 3.1 STFPM

STFPM은 동일한 backbone 구조의 teacher와 student에서 여러 계층의 feature를 추출한다. teacher는 로컬 pretrained 가중치를 사용하고 고정되며, student는 정상 이미지에서 teacher feature를 재현하도록 학습된다.

- 모델 factory: `stfpm_anomaly`
- Adapter: `stfpm`
- Config: `configs/anomaly/models/stfpm.yaml`
- 기본 backbone: `resnet18`
- Selector: `--model.backbone resnet18|resnet50|wide_resnet50_2`
- 학습 대상: student network
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: teacher와 student feature의 계층별 차이를 loss와 anomaly map 생성에 사용한다.

### 3.2 EfficientAD

EfficientAD는 고정된 pretrained PDN teacher, 학습 가능한 student, autoencoder를 함께 사용한다. student–teacher loss와 autoencoder 관련 loss를 합산해 학습하며, 학습 전 teacher feature의 channel 통계를 계산하고 validation 과정에서 anomaly map 정규화용 quantile을 보정한다.

- 모델 factory: `efficientad_anomaly`
- Adapter: `efficientad`
- Config: `configs/anomaly/models/efficientad.yaml`
- 기본 크기: `small`
- Selector: `--model.size small|medium`
- 학습 대상: student와 autoencoder
- 로컬 자산: 크기에 맞는 EfficientAD teacher 가중치와 `Imagenette` 학습 이미지
- 구현 특이사항: 논문 설정에 따라 `batch_size=1`만 허용하며 입력 transform에 `Normalize`를 두지 않는다. 모델 내부에서 ImageNet normalization을 수행한다.
- Calibration 요구사항: `valid` split에 `label == 0`인 정상 샘플이 있어야 한다.

### 3.3 FastFlow

FastFlow는 고정된 pretrained backbone의 feature map 뒤에 2D normalizing-flow block을 연결한다. 학습 시 flow의 hidden variable과 Jacobian으로 loss를 계산하며, backbone은 gradient 계산과 optimizer 대상에서 제외된다.

- 모델 factory: `fastflow_anomaly`
- Adapter: `fastflow`
- Config: `configs/anomaly/models/fastflow.yaml`
- 기본 backbone: `resnet18`
- Selector: `--model.backbone resnet18|wide_resnet50_2`
- 학습 대상: normalizing-flow block
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: `model.params.input_size`는 data config의 `data.image_size`와 일치해야 한다. LayerNorm과 flow block이 고정된 공간 크기로 생성되기 때문이다.

### 3.4 PatchCore

PatchCore는 고정된 pretrained backbone에서 정상 이미지의 patch embedding을 수집한다. 수집한 embedding에 coreset subsampling을 적용해 memory bank를 만들고, 추론 시 최근접 이웃 거리를 이용해 anomaly score와 anomaly map을 생성한다.

- 모델 factory: `patchcore_anomaly`
- Adapter: `patchcore`
- Config: `configs/anomaly/models/patchcore.yaml`
- 기본 backbone: `wide_resnet50_2`
- Selector: `--model.backbone wide_resnet50_2|resnet18`
- 학습 대상: 없음
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: 기본 학습 epoch는 `1`이며 첫 validation 전에 `coreset_sampling_ratio`에 따라 memory bank를 한 번 구축한다.
- Optimizer 특이사항: 공통 engine contract를 충족하기 위해 config에 optimizer가 선언되어 있지만 backbone forward가 `torch.no_grad()`에서 실행되므로 파라미터를 변경하지 않는다.
- 자원 특성: embedding 수집과 coreset 생성 과정의 메모리 사용량은 학습 이미지 수, feature 크기, `coreset_sampling_ratio`의 영향을 받는다.

## 4. 공통 통합 구조

### 4.1 모델 코드와 SSOT

`src/tasks/anomaly/models/` 아래의 모델 구현은 anomalib pure-PyTorch 코드를 SSOT로 사용한다. 통합에 필요한 registry 등록, 로컬 가중치 주입, offline 강제와 lifecycle 차이는 factory 또는 adapter에서 처리한다. 모델 네트워크 구조와 연산 순서를 boilerplate에 맞추기 위해 변경하지 않는다.

### 4.2 Registry와 Adapter

각 model config의 `model.name`은 `MODELS` registry의 factory를 선택하고 `adapter.name`은 `ADAPTERS` registry의 adapter를 선택한다. 공통 engine에는 모델명이나 anomaly task에 따른 조건 분기를 추가하지 않는다.

Adapter가 담당하는 모델별 동작은 다음과 같다.

| 모델 | Adapter의 주요 책임 |
|---|---|
| STFPM | teacher–student feature loss 계산 |
| EfficientAD | 보조 Imagenette loader 구성, teacher 통계 계산, quantile calibration, 복합 loss 계산 |
| FastFlow | flow hidden variable과 Jacobian 기반 loss 계산 |
| PatchCore | embedding 수집과 validation 전 coreset memory bank 구축 |

### 4.3 Offline 실행

모델 생성 과정에서 pretrained 가중치를 자동으로 내려받지 않는다. 가중치와 보조 데이터는 `configs/local.yaml` 또는 환경변수로 지정한 root 아래에 사용자가 준비해야 한다. 필요한 파일이나 폴더가 없으면 실행을 계속하지 않고 오류로 종료한다.

## 5. 설치 및 로컬 자산 요구사항

### 5.1 Config 기준 기본 자산

| 모델 | 기본 경로 |
|---|---|
| STFPM | `${paths.backbone_root}/resnet18-f37072fd.pth` |
| EfficientAD small | `${paths.backbone_root}/efficientad_pretrained_weights/pretrained_teacher_small.pth` |
| EfficientAD medium | `${paths.backbone_root}/efficientad_pretrained_weights/pretrained_teacher_medium.pth` |
| EfficientAD auxiliary data | `${paths.dataset_root}/imagenette2/train` |
| FastFlow | `${paths.backbone_root}/resnet18-f37072fd.pth` |
| PatchCore | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` |

Selector로 backbone이나 모델 크기를 변경하면 해당 selector가 지정하는 가중치 경로도 함께 적용된다. 경로는 config placeholder를 유지하고 제품 코드에 하드코딩하지 않는다.

### 5.2 Python 패키지 특이사항

모델 코드가 직접 사용하는 주요 패키지는 다음과 같다.

| 적용 모델 | 주요 패키지 |
|---|---|
| 공통 | `torch`, `torchvision` |
| STFPM | `timm` |
| EfficientAD | `torchvision` |
| FastFlow | `timm`, `FrEIA`, `scipy`, `omegaconf` |
| PatchCore | `timm`, `kornia`, `scikit-learn`, `tqdm` |

현재 `requirements.txt`에는 일부 모델 패키지가 명시되어 있지 않다. 실행 전 `pytorch_env`에 대상 모델의 패키지가 준비되어 있는지 확인해야 하며, 프로젝트 규칙에 따라 실행 중 자동 설치는 수행하지 않는다.

## 6. 모델 config 선택

학습, 평가, 추론 CLI에는 사용할 모델의 config 경로를 전달한다.

```bash
--model configs/anomaly/models/stfpm.yaml
--model configs/anomaly/models/efficientad.yaml
--model configs/anomaly/models/fastflow.yaml
--model configs/anomaly/models/patchcore.yaml
```

전체 명령어와 selector, override 사용법은 [CLI 사용 가이드](./cli-usage.md)를 참고한다.

## 7. 문서 갱신 기준

`/add-anomalib-model`로 모델을 추가할 때 이 문서도 다음 순서로 갱신한다.

1. 전체 구현 현황 표에 모델과 대표 패러다임을 추가한다.
2. 모델 분류 체계의 해당 패러다임에 모델을 추가한다.
3. 모델 factory, adapter, config, 학습 방식, 로컬 자산을 모델별 구현 개요에 기록한다.
4. 추가 Python 패키지와 설치 특이사항을 기록한다.
5. 개별 모델 문서가 작성되면 모델별 구현 개요에서 연결한다.

성능 수치, 데이터셋별 benchmark, 모델 간 정확도 비교는 이 문서에 기록하지 않는다. 해당 평가는 사용자가 별도로 수행하고 관리한다.

## 8. 개별 모델 문서

개별 모델의 상세 문서는 추후 별도로 작성한다. 이 문서는 전체 모델 목록과 구현 상태를 확인하는 index 역할을 유지한다.

## 9. 참고 자료

- [프로젝트 폴더 구조 가이드](./structure.md)
- [CLI 사용 가이드](./cli-usage.md)
- `docs/dev/v0.2/BRIEF.md`
