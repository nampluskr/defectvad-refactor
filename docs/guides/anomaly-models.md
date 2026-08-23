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
| Feature Embedding / Memory Bank | PaDiM | 패치별 다변량 가우시안 분포 | 구현됨 | 지원 | 지원 | 지원 | `resnet18` | backbone 가중치 |
| Teacher–Student / Knowledge Distillation | Reverse Distillation | 역방향 teacher–student feature 재구성 | 구현됨 | 지원 | 지원 | 지원 | `wide_resnet50_2` | backbone 가중치 |
| Density Estimation | DFM | PCA 재구성 오차 또는 가우시안 NLL | 구현됨 | 지원 | 지원 | 지원 | `resnet50` | backbone 가중치 |
| Density Estimation | DFKDE | PCA + Gaussian KDE (image-level만) | 구현됨 | 지원 | 지원(image-level) | 지원 | `resnet18` | backbone 가중치 |
| Feature Embedding / Memory Bank | CFA | 좌표 인지 클러스터 중심까지의 거리 (gradient로 학습) | 구현됨 | 지원 | 지원 | 지원 | `wide_resnet50_2` | backbone 가중치 |

아홉 모델은 `src/tasks/anomaly/models/` 아래의 pure-PyTorch 모델과 `src/tasks/anomaly/adapters/` 아래의 lifecycle adapter로 구성된다. 학습 열의 `지원`은 gradient 학습만을 뜻하지 않는다. PatchCore의 학습 단계는 파라미터 최적화 대신 정상 이미지의 embedding을 수집하고 memory bank를 구축한다. DFKDE는 이미지 단위 anomaly score만 산출하므로 평가·시각화에서 pixel 단위 지표를 제공하지 않는다.

## 2. 모델 분류 체계

모델은 anomaly score를 만드는 주된 방식을 기준으로 하나의 대표 패러다임에 배치한다. 여러 방식을 결합한 모델은 보조 패러다임을 함께 표기한다.

### 2.1 Teacher–Student / Knowledge Distillation

고정된 teacher가 추출한 정상 feature를 student가 재현하도록 학습하고 두 출력의 차이를 anomaly signal로 사용한다.

- STFPM
- EfficientAD
- Reverse Distillation

EfficientAD는 teacher–student 구조와 autoencoder 기반 reconstruction을 결합하므로 `Reconstruction`을 보조 패러다임으로 함께 표기한다. Reverse Distillation은 student가 teacher feature를 그대로 재현하는 대신 teacher의 bottleneck 표현에서 원본 feature 피라미드를 역으로 재구성하도록 학습한다.

### 2.2 Normalizing Flow

사전 학습된 backbone feature를 invertible flow로 변환하여 정상 feature의 분포를 학습한다.

- FastFlow

### 2.3 Feature Embedding / Memory Bank

정상 이미지의 patch embedding을 memory bank에 저장하고 입력 patch와 가까운 정상 embedding 사이의 거리를 anomaly signal로 사용한다.

- PatchCore
- PaDiM
- CFA

PaDiM은 개별 embedding을 memory bank에 그대로 저장하는 대신 patch 위치별로 embedding 분포를 다변량 가우시안으로 요약해 저장하고, 추론 시 Mahalanobis distance를 anomaly signal로 사용한다. CFA는 memory bank가 정상 embedding의 (선택적으로 k-means로 축소한) 클러스터 중심으로 구성되고, 이 중심까지의 거리를 gradient로 직접 학습하는 descriptor network를 통해 최소화한다는 점에서 PatchCore·PaDiM과 다르다 — memory bank가 고정된 통계가 아니라 학습 대상이다.

### 2.4 Density Estimation

정상 이미지에서 추출한 backbone feature 분포를 별도의 밀도 모델(PCA 재구성, Gaussian, KDE)로 추정하고, 새 샘플이 그 분포에서 벗어난 정도를 anomaly signal로 사용한다.

- DFM
- DFKDE

DFM은 `score_type` 설정에 따라 PCA 재구성 오차(`fre`, pixel-level 지원)와 가우시안 음의 로그가능도(`nll`, image-level 전용) 중 하나를 사용한다. DFKDE는 PCA로 축소한 feature에 non-parametric Gaussian KDE를 적합시키며, 항상 image-level score만 산출한다.

## 3. 모델별 구현 개요

아래 파라미터 표는 model factory에 전달되는 `model.params`만 다룬다. `지원값`에는 factory가 받을 수 있는 값이나 형식과 현재 config selector로 제공되는 값을 구분해 기록한다. Adapter, optimizer, scheduler 파라미터는 포함하지 않는다.

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

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `resnet18` | 호환되는 `timm` backbone 이름. Config selector: `resnet18`, `resnet50`, `wide_resnet50_2` | teacher와 student가 공통으로 사용하는 feature extraction backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/resnet18-f37072fd.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | teacher backbone에 적용할 pretrained 가중치 경로 |
| `layers` | `list[str] \| tuple[str, ...]` | `["layer1", "layer2", "layer3"]` | 선택한 backbone에서 제공하는 중간 계층 이름 | feature distillation과 anomaly map 생성에 사용할 계층 목록 |

`--model.backbone` selector를 사용하면 `backbone`과 `weights_path`가 함께 변경된다. `layers`는 현재 config에 직접 선언되지 않아 factory 기본값을 사용한다.

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

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `weights_path` | `str \| None` | `${paths.backbone_root}/efficientad_pretrained_weights/pretrained_teacher_small.pth` | 선택한 모델 크기와 일치하는 로컬 파일 경로 | pretrained PDN teacher 가중치 경로 |
| `teacher_out_channels` | `int` | `384` | 양의 정수 | teacher와 student가 출력하는 feature channel 수 |
| `model_size` | `str` | `small` | `small`, `medium` | PDN teacher와 student의 네트워크 크기 |
| `padding` | `bool` | `false` | `true`, `false` | PDN convolution에 padding을 적용할지 여부 |
| `pad_maps` | `bool` | `true` | `true`, `false` | 출력 anomaly map의 가장자리를 입력 크기에 맞게 보정할지 여부 |

`--model.size` selector를 사용하면 `model_size`와 `weights_path`가 함께 변경된다. `teacher_out_channels` 또는 구조 관련 값을 변경하면 제공된 teacher 가중치와 호환되지 않을 수 있다.

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

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `resnet18` | Factory: `resnet18`, `wide_resnet50_2`, `cait_m48_448`, `deit_base_distilled_patch16_384`. Config selector: `resnet18`, `wide_resnet50_2` | 고정된 feature extractor로 사용할 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/resnet18-f37072fd.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `input_size` | `list[int] \| tuple[int, int]` | `[256, 256]` | `[height, width]` 형식의 양의 정수 쌍 | LayerNorm과 normalizing-flow block을 구성할 입력 이미지 크기 |
| `flow_steps` | `int` | `8` | 양의 정수 | feature scale마다 순차적으로 구성할 flow block 수 |
| `conv3x3_only` | `bool` | `false` | `true`, `false` | flow subnet에서 모든 convolution kernel을 3×3으로 제한할지 여부 |
| `hidden_ratio` | `float` | `1.0` | 양수 | flow subnet의 hidden channel 크기를 결정하는 비율 |

`--model.backbone` selector를 사용하면 `backbone`과 `weights_path`가 함께 변경된다. Transformer backbone은 selector로 제공되지 않으므로 사용할 경우 `backbone`, `weights_path`, `input_size`를 함께 지정해야 한다. `input_size`는 반드시 data config의 `data.image_size`와 같아야 한다.

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

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `wide_resnet50_2` | 호환되는 `timm` backbone 이름. Config selector: `wide_resnet50_2`, `resnet18` | patch feature를 추출할 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `layers` | `list[str] \| tuple[str, ...]` | `["layer2", "layer3"]` | 선택한 backbone에서 제공하는 중간 계층 이름 | patch embedding 생성에 사용할 feature 계층 목록 |
| `num_neighbors` | `int` | `9` | memory bank 크기 이하의 양의 정수 | anomaly score 계산에 사용할 최근접 memory-bank embedding 수 |

`--model.backbone` selector를 사용하면 `backbone`과 `weights_path`가 함께 변경된다. `coreset_sampling_ratio`는 모델 factory가 아니라 `adapter.params`에 속하므로 이 표에서 제외한다.

### 3.5 PaDiM

PaDiM은 고정된 pretrained backbone에서 정상 이미지의 patch embedding을 수집한다. patch 위치별로 embedding 채널을 무작위 부분집합으로 축소한 뒤, 위치마다 다변량 가우시안 분포를 적합시켜 평균과 공분산을 memory bank로 저장한다. 추론 시 각 위치의 embedding과 해당 가우시안 사이의 Mahalanobis distance로 anomaly map을 만든다.

- 모델 factory: `padim_anomaly`
- Adapter: `padim`
- Config: `configs/anomaly/models/padim.yaml`
- 기본 backbone: `resnet18`
- Selector: `--model.backbone resnet18|wide_resnet50_2`
- 학습 대상: 없음
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: 기본 학습 epoch는 `1`이며 첫 validation 전에 수집한 embedding으로 위치별 다변량 가우시안을 한 번 적합시킨다. anomaly map 생성 내부에서 `GaussianBlur2d(sigma=4)`를 이미 적용하므로 adapter의 `smooth_sigma`는 `0`으로 고정한다.
- Optimizer 특이사항: 공통 engine contract를 충족하기 위해 config에 optimizer가 선언되어 있지만 backbone forward가 `torch.no_grad()`에서 실행되므로 파라미터를 변경하지 않는다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `resnet18` | 호환되는 `timm` backbone 이름. Config selector: `resnet18`, `wide_resnet50_2` | patch feature를 추출할 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/resnet18-f37072fd.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `layers` | `list[str] \| tuple[str, ...]` | `["layer1", "layer2", "layer3"]` | 선택한 backbone에서 제공하는 중간 계층 이름 | patch embedding 생성에 사용할 feature 계층 목록 |
| `n_features` | `int \| None` | `null` (factory가 backbone별 논문 기본값 사용) | `0 < n_features <= backbone의 원본 embedding 채널 수`. Config selector: `resnet18` → `100`, `wide_resnet50_2` → `550` | 차원 축소 후 유지할 embedding 채널 수 |

`--model.backbone` selector를 사용하면 `backbone`, `weights_path`, `n_features`가 함께 변경된다. `n_features`를 selector 없이 직접 지정할 경우 backbone의 원본 embedding 채널 수를 초과할 수 없다.

### 3.6 Reverse Distillation

Reverse Distillation은 고정된 pretrained encoder(teacher)의 다중 계층 feature를 bottleneck에서 하나의 압축 표현으로 합친 뒤, decoder(student)가 이 압축 표현으로부터 encoder의 원본 feature 피라미드를 역으로 재구성하도록 학습한다. encoder와 decoder 사이의 계층별 feature 차이를 anomaly signal로 사용한다.

- 모델 factory: `reverse_distillation_anomaly`
- Adapter: `reverse_distillation`
- Config: `configs/anomaly/models/reverse_distillation.yaml`
- 기본 backbone: `wide_resnet50_2`
- Selector: `--model.backbone wide_resnet50_2|resnet18|resnet50`, `--model.mode add|multiply`
- 학습 대상: bottleneck과 decoder
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: encoder는 factory에서 고정되고 `forward`가 매 호출마다 `eval()`을 강제하므로 engine의 epoch마다의 `model.train()`으로도 학습 모드로 전환되지 않는다. anomaly map 생성 내부에서 `GaussianBlur2d(sigma=4)`를 이미 적용하므로 adapter의 `smooth_sigma`는 `0`으로 고정한다.
- Optimizer 특이사항: anomalib은 decoder와 bottleneck 파라미터에만 `Adam(lr=0.005, betas=(0.5, 0.99))`를 적용한다. 이 프로젝트는 factory가 encoder를 미리 고정해 두므로 공통 single-group optimizer builder를 그대로 사용해도 동일한 파라미터 집합이 선택된다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `wide_resnet50_2` | 호환되는 `timm` backbone 이름. Config selector: `wide_resnet50_2`, `resnet18`, `resnet50` | encoder(teacher)와 decoder(student) 구조를 결정하는 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | encoder에 적용할 pretrained 가중치 경로 |
| `layers` | `list[str] \| tuple[str, ...]` | `["layer1", "layer2", "layer3"]` | 선택한 backbone에서 제공하는 중간 계층 이름 | encoder–decoder feature 비교와 anomaly map 생성에 사용할 계층 목록 |
| `input_size` | `list[int] \| tuple[int, int]` | `[256, 256]` | `[height, width]` 형식의 양의 정수 쌍 | anomaly map 생성 캔버스를 고정할 입력 이미지 크기 |
| `anomaly_map_mode` | `str` | `add` | `add`, `multiply`. Config selector: `add`, `multiply` | 계층별 anomaly map을 합성하는 방식 |

`--model.backbone` selector를 사용하면 `backbone`과 `weights_path`가 함께 변경된다. `input_size`는 반드시 data config의 `data.image_size`와 같아야 한다. `anomaly_map_mode`는 `--model.mode` selector로 독립적으로 변경할 수 있다.

### 3.7 DFM

DFM은 고정된 pretrained backbone에서 단일 계층의 pooled feature를 수집한다. 수집한 feature에 PCA를 적합시키고, `score_type`에 따라 PCA 재구성 오차(`fre`) 또는 재구성된 feature에 적합시킨 단일 가우시안의 음의 로그가능도(`nll`)를 anomaly score로 사용한다.

- 모델 factory: `dfm_anomaly`
- Adapter: `dfm`
- Config: `configs/anomaly/models/dfm.yaml`
- 기본 backbone: `resnet50`
- Selector: `--model.backbone resnet50|resnet18|wide_resnet50_2`
- 학습 대상: 없음
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: 기본 학습 epoch는 `1`이며 첫 validation 전에 수집한 feature로 PCA(및 `nll`일 때 가우시안)를 한 번 적합시킨다. `fre` 모드의 anomaly map은 내부에서 blur를 적용하지 않으므로 PatchCore·PaDiM·Reverse Distillation과 달리 adapter의 `smooth_sigma`는 공통 기본값(`4.0`)을 유지한다. `nll` 모드는 anomaly map을 생성하지 않으며, 이 프로젝트의 공통 pixel-metric pipeline과 연결되어 있지 않다.
- Optimizer 특이사항: 공통 engine contract를 충족하기 위해 config에 optimizer가 선언되어 있지만 backbone forward가 `torch.no_grad()`에서 실행되므로 파라미터를 변경하지 않는다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `resnet50` | 호환되는 `timm` backbone 이름. Config selector: `resnet50`, `resnet18`, `wide_resnet50_2` | feature를 추출할 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/resnet50-0676ba61.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `layer` | `str` | `layer3` | 선택한 backbone에서 제공하는 단일 중간 계층 이름 | feature를 추출할 단일 계층 (PatchCore·PaDiM과 달리 계층이 하나뿐이다) |
| `pooling_kernel_size` | `int` | `4` | 양의 정수 | 추출한 feature map에 적용할 average pooling kernel 크기 |
| `n_comps` | `float` | `0.97` | `0 < n_comps <= 1`이면 보존할 분산 비율, `1` 초과 정수면 정확한 component 수 | PCA가 유지할 component 수 또는 분산 비율 |
| `score_type` | `str` | `fre` | `fre` (pixel-level 지원), `nll` (image-level 전용) | anomaly score 계산 방식 |

`--model.backbone` selector를 사용하면 `backbone`과 `weights_path`가 함께 변경된다. `layer`는 selector로 제공되지 않으므로 backbone을 바꿀 때 유효한 계층 이름인지 직접 확인해야 한다.

### 3.8 DFKDE

DFKDE는 고정된 pretrained backbone에서 여러 계층의 feature를 global average pooling으로 압축해 하나의 벡터로 이어 붙인다. 수집한 벡터에 PCA로 차원을 축소한 뒤 non-parametric Gaussian KDE를 적합시키고, 새 샘플의 KDE 밀도를 sigmoid로 변환한 값을 anomaly score로 사용한다.

- 모델 factory: `dfkde_anomaly`
- Adapter: `dfkde`
- Config: `configs/anomaly/models/dfkde.yaml`
- 기본 backbone: `resnet18`
- Selector: `--model.backbone resnet18|wide_resnet50_2`
- 학습 대상: 없음
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: DFKDE는 anomaly map을 생성하지 않는 이 프로젝트의 유일한 모델이다. `DfkdeAdapter`가 공통 `eval_step`·`predict_step`·threshold 보정을 재정의해 pixel 단위 로직을 건너뛴다. model config가 `metrics:`를 `image_auroc` 하나로 재정의해 `pixel_auroc`를 제거한다 — data config는 건드리지 않는다.
- Optimizer 특이사항: 공통 engine contract를 충족하기 위해 config에 optimizer가 선언되어 있지만 backbone forward가 `torch.no_grad()`에서 실행되므로 파라미터를 변경하지 않는다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `resnet18` | 호환되는 `timm` backbone 이름. Config selector: `resnet18`, `wide_resnet50_2` | feature를 추출할 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/resnet18-f37072fd.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `layers` | `list[str] \| tuple[str, ...]` | `["layer4"]` | 선택한 backbone에서 제공하는 중간 계층 이름 | global pooling 후 이어 붙일 feature 계층 목록 |
| `n_pca_components` | `int` | `16` | 양의 정수 | KDE 이전 차원 축소 단계에서 유지할 PCA component 수 |
| `feature_scaling_method` | `str` | `scale` | `norm` (L2 정규화), `scale` (학습 시 관측된 최대 길이로 정규화) | KDE 이전 feature 스케일링 방식 |
| `max_training_points` | `int` | `40000` | 양의 정수 | KDE 적합에 사용할 최대 샘플 수 (초과 시 무작위 부분집합 사용) |

`--model.backbone` selector를 사용하면 `backbone`과 `weights_path`가 함께 변경된다.

### 3.9 CFA

CFA는 고정된 pretrained backbone의 다중 계층 feature를 CoordConv 기반 descriptor network로 변환한다. 학습 시작 전 전체 학습셋에 대해 한 번 forward pass를 수행해 정상 feature의 (선택적으로 k-means로 축소한) 클러스터 중심을 memory bank로 초기화하고, 이후 descriptor network가 이 중심까지의 거리를 hypersphere 손실로 gradient 학습한다. PatchCore·PaDiM과 달리 memory bank 자체가 아니라 그 중심까지의 거리를 좁히는 network를 학습한다는 점이 다르다.

- 모델 factory: `cfa_anomaly`
- Adapter: `cfa`
- Config: `configs/anomaly/models/cfa.yaml`
- 기본 backbone: `wide_resnet50_2`
- Selector: `--model.backbone wide_resnet50_2|resnet18`
- 학습 대상: descriptor network (CoordConv2d)
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: `CfaModel`은 `TimmFeatureExtractor`가 아니라 `torchvision.models.feature_extraction.create_feature_extractor`로 backbone을 감싼다. 팩토리가 구성 중 `torchvision.models.<backbone>`을 `pretrained=False` wrapper로 일시 치환해 오프라인을 유지한다. 학습 시작 전 `CfaAdapter.on_fit_start`가 이 프로젝트의 `(images, targets)` 배치를 `initialize_centroid`가 기대하는 `.image` 속성 객체로 감싸 전달한다 — 모델 파일은 수정하지 않는다. anomaly map 생성 내부에서 `GaussianBlur2d(sigma=4)`를 이미 적용하므로 adapter의 `smooth_sigma`는 `0`으로 고정한다.
- Optimizer 특이사항: 팩토리가 backbone을 `requires_grad=False`로 고정하므로 공통 single-group optimizer builder는 descriptor network 파라미터만 선택한다 — anomalib 원본은 `model.parameters()` 전체를 optimizer에 넘기지만 backbone forward가 항상 `torch.no_grad()`에서 실행되어 그레이디언트가 도달하지 않으므로 결과는 동일하다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `wide_resnet50_2` | Factory: `resnet18`, `wide_resnet50_2`, `vgg19_bn` (`efficientnet_b5`는 미구현). Config selector: `wide_resnet50_2`, `resnet18` | descriptor network 입력으로 사용할 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `gamma_c` | `int` | `1` | `1` 이상의 정수 | `1` 초과면 memory bank를 k-means로 `(H×W)/gamma_c`개 클러스터 중심으로 축소 |
| `gamma_d` | `int` | `1` | 양의 정수 | descriptor network 출력 채널을 `backbone 차원 / gamma_d`로 축소 |
| `num_nearest_neighbors` | `int` | `3` | 양의 정수 | anomaly map 계산 시 사용할 최근접 이웃 수 |
| `num_hard_negative_features` | `int` | `3` | 양의 정수 | attraction/repulsion 손실 계산에 사용할 hard negative feature 수 |
| `radius` | `float` | `1e-5` | 양수 | hypersphere 결정 경계의 초기 반지름. gradient로 학습되지 않는 고정 상수 |

`--model.backbone` selector를 사용하면 `backbone`과 `weights_path`가 함께 변경된다. `num_nearest_neighbors`·`num_hard_negative_features`·`radius`는 `model.params`에만 선언한다 — `CfaAdapter.on_fit_start`가 이 값들을 `adapter.params`에 별도로 복제하는 대신 생성된 `CfaModel` 인스턴스에서 직접 읽어 `CfaLoss`를 구성하므로, anomaly map과 학습 손실이 서로 다른 하이퍼파라미터를 쓰게 될 여지가 없다.

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
| PaDiM | embedding 수집과 validation 전 위치별 다변량 가우시안 적합 |
| Reverse Distillation | encoder–decoder feature 간 cosine-similarity loss 계산 |
| DFM | feature 수집과 validation 전 PCA(및 nll 모드의 가우시안) 적합 |
| DFKDE | feature 수집, validation 전 PCA+KDE 적합, image-level 전용 eval/predict/threshold 재정의 |
| CFA | 학습 시작 전 memory bank 중심 초기화, hypersphere 손실 계산 |

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
| PaDiM | `${paths.backbone_root}/resnet18-f37072fd.pth` |
| Reverse Distillation | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` |
| DFM | `${paths.backbone_root}/resnet50-0676ba61.pth` |
| DFKDE | `${paths.backbone_root}/resnet18-f37072fd.pth` |
| CFA | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` |

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
| PaDiM | `timm` |
| Reverse Distillation | `timm` |
| DFM | `timm` |
| DFKDE | `timm` |
| CFA | `torchvision`(feature extraction), `einops`, `scikit-learn`(`gamma_c > 1`일 때 k-means) |

현재 `requirements.txt`에는 일부 모델 패키지가 명시되어 있지 않다. 실행 전 `pytorch_env`에 대상 모델의 패키지가 준비되어 있는지 확인해야 하며, 프로젝트 규칙에 따라 실행 중 자동 설치는 수행하지 않는다.

## 6. 모델 config 선택

학습, 평가, 추론 CLI에는 사용할 모델의 config 경로를 전달한다.

```bash
--model configs/anomaly/models/stfpm.yaml
--model configs/anomaly/models/efficientad.yaml
--model configs/anomaly/models/fastflow.yaml
--model configs/anomaly/models/patchcore.yaml
--model configs/anomaly/models/padim.yaml
--model configs/anomaly/models/reverse_distillation.yaml
--model configs/anomaly/models/dfm.yaml
--model configs/anomaly/models/dfkde.yaml
--model configs/anomaly/models/cfa.yaml
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
