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
| Normalizing Flow | CFLOW | fiber(feature-vector 조각) 단위 conditional normalizing flow | 구현됨 | 지원 (adapter 소유 private optimizer, §4.2) | 지원 | 지원 | `wide_resnet50_2` | backbone 가중치 |
| Autoencoder / Reconstruction | FRE | Tied AutoEncoder 기반 feature reconstruction error | 구현됨 | 지원 | 지원 | 지원 | `resnet50` | backbone 가중치 |
| Normalizing Flow | U-Flow | U자형 다중 스케일 normalizing flow | 구현됨 | 지원 | 지원 | 지원 | `resnet18` | backbone 가중치 |
| Normalizing Flow | CS-Flow | Cross-scale coupling normalizing flow | 구현됨 | 지원 | 지원 | 지원 | `efficientnet_b5` | backbone 가중치 |
| Discriminative / Synthetic Anomaly | SuperSimpleNet | Feature adaptation 및 Perlin noise 합성 기반 segmentation-detection | 구현됨 | 지원 | 지원 | 지원 | `wide_resnet50_2` | backbone 가중치 |
| Reconstruction / Adversarial | GANomaly | Generator-Discriminator 적대적 생성 및 잠재 공간 오차 (image-level) | 구현됨 | 지원 (Dual Adam optimizer) | 지원(image-level) | 지원 | 자체 합성곱 인코더/디코더 | 없음 (from scratch) |
| Reconstruction / Discriminative | DRAEM | 재구성-판별 듀얼 서브네트워크 + DTD/Perlin 합성 인공 결함 지도학습 | 구현됨 | 지원 | 지원 | 지원 | 자체 서브네트워크 | DTD 텍스처 데이터셋 |
| Reconstruction / Discrete Latent | DSR | VQ-VAE 이산 코드북 + 듀얼 서브스페이스 재투영 + 2단계 다중 옵티마이저 학습 | 구현됨 | 지원 (2단계 다중 옵티마이저) | 지원 | 지원 | VQ-VAE 이산 모델 | VQ-VAE pretrained 가중치 |
| Teacher–Student / Foundation Model 무관 | UniNet | source/target 이중 teacher + attention bottleneck + DFS | 구현됨 | 지원 (split-LR AdamW) | 지원 | 지원 | `wide_resnet50_2` | backbone 가중치 |
| Reconstruction / DINOv2 ViT | Dinomaly | frozen DINOv2 ViT encoder + bottleneck/decoder feature 재구성 | 구현됨 | 지원 (adapter 소유 private optimizer/scheduler, §4.2) | 지원 | 지원 | `dinov2reg_vit_base_14` | DINOv2 pretrained 가중치 |
| Feature Embedding / Memory Bank (DINOv2 ViT) | AnomalyDINO | DINOv2 ViT patch feature memory bank, no-gradient | 구현됨 | 지원 (fit-only, 1 epoch로 완결) | 지원 | 지원 | `dinov2_vit_small_14` | DINOv2 pretrained 가중치 |

스무 모델은 `src/tasks/anomaly/models/` 아래의 pure-PyTorch 모델과 `src/tasks/anomaly/adapters/` 아래의 lifecycle adapter로 구성된다. 학습 열의 `지원`은 gradient 학습만을 뜻하지 않는다. PatchCore의 학습 단계는 파라미터 최적화 대신 정상 이미지의 embedding을 수집하고 memory bank를 구축한다. DFKDE와 GANomaly는 이미지 단위 anomaly score만 산출하므로 평가·시각화에서 pixel 단위 지표를 제공하지 않는다.

## 2. 모델 분류 체계

모델은 anomaly score를 만드는 주된 방식을 기준으로 하나의 대표 패러다임에 배치한다. 여러 방식을 결합한 모델은 보조 패러다임을 함께 표기한다.

### 2.1 Teacher–Student / Knowledge Distillation

고정된 teacher가 추출한 정상 feature를 student가 재현하도록 학습하고 두 출력의 차이를 anomaly signal로 사용한다.

- STFPM
- EfficientAD
- Reverse Distillation
- UniNet

EfficientAD는 teacher–student 구조와 autoencoder 기반 reconstruction을 결합하므로 `Reconstruction`을 보조 패러다임으로 함께 표기한다. Reverse Distillation은 student가 teacher feature를 그대로 재현하는 대신 teacher의 bottleneck 표현에서 원본 feature 피라미드를 역으로 재구성하도록 학습한다. UniNet은 고정된 source teacher와 학습 가능한 target teacher 이중 구조를 attention bottleneck과 DFS(Domain-related Feature Selection)로 결합해, student가 두 teacher의 feature를 동시에 근사하도록 학습한다.

### 2.2 Normalizing Flow

사전 학습된 backbone feature를 invertible flow로 변환하여 정상 feature의 분포를 학습한다.

- FastFlow
- CFLOW
- U-Flow
- CS-Flow

FastFlow는 backbone feature map 전체를 2D spatial flow로 한 번에 변환한다. CFLOW는 feature map을 위치별 벡터("fiber")로 펼쳐 위치 정보로 조건화한 뒤 fiber 단위로 flow를 학습한다. U-Flow는 U자형 아키텍처를 기반으로 서로 다른 스케일의 feature를 역전파 가능한 flow 블록과 결합하여 다중 스케일 밀도를 학습한다. CS-Flow는 3개 스케일 간 상호 연결된 cross-scale coupling convolution layer를 통해 다중 해상도 feature의 결합 분포를 학습한다.

### 2.3 Feature Embedding / Memory Bank

정상 이미지의 patch embedding을 memory bank에 저장하고 입력 patch와 가까운 정상 embedding 사이의 거리를 anomaly signal로 사용한다.

- PatchCore
- PaDiM
- CFA
- AnomalyDINO

PaDiM은 개별 embedding을 memory bank에 그대로 저장하는 대신 patch 위치별로 embedding 분포를 다변량 가우시안으로 요약해 저장하고, 추론 시 Mahalanobis distance를 anomaly signal로 사용한다. CFA는 memory bank가 정상 embedding의 (선택적으로 k-means로 축소한) 클러스터 중심으로 구성되고, 이 중심까지의 거리를 gradient로 직접 학습하는 descriptor network를 통해 최소화한다는 점에서 PatchCore·PaDiM과 다르다 — memory bank가 고정된 통계가 아니라 학습 대상이다. AnomalyDINO는 PatchCore와 동일한 memory bank/coreset 메커니즘(`KCenterGreedy`, `AnomalyMapGenerator`)을 재사용하되, backbone을 CNN 대신 frozen DINOv2 ViT로 교체하고 patch feature를 L2 정규화 후 코사인 거리로 비교한다.

### 2.4 Density Estimation

정상 이미지에서 추출한 backbone feature 분포를 별도의 밀도 모델(PCA 재구성, Gaussian, KDE)로 추정하고, 새 샘플이 그 분포에서 벗어난 정도를 anomaly signal로 사용한다.

- DFM
- DFKDE

DFM은 `score_type` 설정에 따라 PCA 재구성 오차(`fre`, pixel-level 지원)와 가우시안 음의 로그가능도(`nll`, image-level 전용) 중 하나를 사용한다. DFKDE는 PCA로 축소한 feature에 non-parametric Gaussian KDE를 적합시키며, 항상 image-level score만 산출한다.

### 2.5 Autoencoder / Reconstruction

정상 feature 또는 이미지를 저차원 잠재 공간으로 압축한 뒤 재구성하고, 원본과의 재구성 오차(Reconstruction Error)를 anomaly signal로 사용한다.

- FRE
- Dinomaly

FRE는 고정된 CNN backbone feature를 Tied AutoEncoder(가중치 공유 선형 오토인코더)로 학습하여 feature 재구성 오차(MSE)를 픽셀 단위 이상 맵과 스코어로 계산한다. Dinomaly는 frozen DINOv2 ViT encoder의 다중 계층 feature를 bottleneck MLP로 압축한 뒤 ViT decoder로 재구성하고, encoder-decoder 간 코사인 유사도를 anomaly signal로 사용한다.

### 2.6 Discriminative / Synthetic Anomaly

정상 feature 공간에 Perlin 노이즈와 가우시안 섭동을 합성(pseudo-anomaly)하여 생성하고, segmentation 헤드와 classification 헤드를 통해 정상과 합성 이상을 판별하도록 학습한다.

- SuperSimpleNet

SuperSimpleNet은 사전 학습된 backbone feature를 2배 업스케일링 및 풀링한 뒤, 1x1 projection adapter와 train-time anomaly generator를 통해 생성된 합성 이상 feature를 판별하여 빠른 속도와 높은 분별력을 동시에 확보한다.

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

### 3.10 CFLOW

CFLOW는 고정된 pretrained backbone의 다중 계층 feature를 conditional normalizing flow decoder로 변환해 정상 feature의 log-likelihood를 학습한다. Feature map을 위치별 벡터("fiber")로 펼친 뒤 위치 정보를 2D sinusoidal encoding으로 조건화하고, 레이어마다 독립된 invertible decoder(`FrEIA`의 `AllInOneBlock` 8개 결합)가 이 조건부 분포를 모델링한다. 추론 시에는 각 위치의 log-likelihood를 정상성 점수로 변환해 레이어별 맵을 합산·반전한 뒤 anomaly map을 만든다.

- 모델 factory: `cflow_anomaly`
- Adapter: `cflow`
- Config: `configs/anomaly/models/cflow.yaml`
- 기본 backbone: `wide_resnet50_2`
- Selector: `--model.backbone wide_resnet50_2|resnet18`
- 학습 대상: decoder(정규화 흐름) 네트워크만 — backbone은 고정
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: Upstream은 이미지 배치 하나를 `fiber_batch_size`(기본 64) 단위 조각으로 잘라 각 조각마다 별도로 optimizer step을 수행한다(256×256·batch 8·`wide_resnet50_2` 기준 배치당 약 168회). 이 프로젝트의 공통 engine은 `train_step` 호출당 정확히 1회의 step만 수행하므로, `CflowAdapter`가 decoder 파라미터 전용 `torch.optim.Adam`을 직접 소유하고 fiber마다 upstream과 동일한 순서로 즉시 step한다 — 공통 engine에는 손대지 않는다(§4.2 참조).
- 알려진 한계: `--resume`은 이 private optimizer의 Adam 모멘텀을 보존하지 않는다(모델 가중치·RNG는 정상 복원). `runtime.amp`/`train.grad_clip`은 이 모델의 실제 decoder 업데이트에 영향을 주지 않는다(기본값에서는 무해). 상세는 `docs/dev/v0.2/reports/UPSTREAM-INVENTORY.md` §14, 적대적 검증 기록은 `docs/dev/v0.2/reviews/A2.md` 참조.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `wide_resnet50_2` | Config selector: `wide_resnet50_2`, `resnet18` | feature 추출에 사용할 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `layers` | `list[str]` | `[layer2, layer3, layer4]` | backbone이 제공하는 layer 이름 목록 | feature를 추출할 backbone 레이어 |
| `fiber_batch_size` | `int` | `64` | 양의 정수 | decoder 학습/추론 시 한 번에 처리할 feature-vector 조각 크기 |
| `decoder` | `str` | `freia-cflow` | `freia-cflow` | invertible decoder 아키텍처 종류 |
| `condition_vector` | `int` | `128` | 4의 배수인 양의 정수 | 위치 조건화에 사용할 벡터 길이 |
| `coupling_blocks` | `int` | `8` | 양의 정수 | decoder당 coupling block 개수 |
| `clamp_alpha` | `float` | `1.9` | 양수 | affine coupling의 clamp 값 |
| `permute_soft` | `bool` | `false` | `true`/`false` | `true`면 SO(N) 소프트 순열 사용(고차원에서 느림) |

`adapter.params.lr`(기본 `0.0001`)이 이 모델의 실제 학습률을 결정한다 — `optim.optimizer` config 블록은 공통 engine이 구조적으로 요구하지만 CFLOW에는 비활성이다(§4.2).

### 3.11 FRE

FRE(Feature Reconstruction Error)는 고정된 CNN backbone에서 추출한 feature를 Tied AutoEncoder(선형 가중치 공유 오토인코더)로 재구성하도록 학습한다. 재구성 오차(MSE)를 기반으로 픽셀 단위 이상 맵과 스코어를 산출한다.

- 모델 factory: `fre_anomaly` (별칭: `fre`)
- Adapter: `fre`
- Config: `configs/anomaly/models/fre.yaml`
- 기본 backbone: `resnet50`
- Selector: `--model.backbone resnet50|wide_resnet50_2|resnet18`
- 학습 대상: `TiedAE` (가중치 및 bias 파라미터)
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: feature extraction 단계에서 `pooling_kernel_size`로 avg pooling을 거쳐 고정 차원 feature vector를 형성하고 `MSELoss`로 Tied AutoEncoder를 최적화한다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `resnet50` | Config selector: `resnet50`, `wide_resnet50_2`, `resnet18` | feature 추출에 사용할 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/resnet50-0676ba61.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `layer` | `str` | `layer3` | backbone이 제공하는 layer 이름 | feature를 추출할 backbone 레이어 |
| `pooling_kernel_size` | `int` | `2` | 양의 정수 (ResNet50: 2, Wide ResNet50: 4) | 추출된 feature map에 적용할 average pooling 커널 크기 |
| `input_dim` | `int` | `65536` | 양의 정수 (pooling 후 채널x높이x너비) | Tied AutoEncoder의 입력 차원 |
| `latent_dim` | `int` | `220` | 양의 정수 | Tied AutoEncoder의 압축 잠재 공간 차원 |

### 3.12 U-Flow

U-Flow는 U자형 normalizing flow 아키텍처를 사용하여 CNN backbone(ResNet 또는 CaiT)에서 추출한 다중 스케일 feature의 결합 확률 밀도를 학습한다. 학습 시 log-likelihood와 log-Jacobian determinant 손실을 결합하고, 추론 시 다중 스케일 likelihood를 평균하여 이상 맵과 스코어를 산출한다.

- 모델 factory: `uflow_anomaly` (별칭: `uflow`)
- Adapter: `uflow`
- Config: `configs/anomaly/models/uflow.yaml`
- 기본 backbone: `resnet18`
- Selector: `--model.backbone resnet18|wide_resnet50_2`
- 학습 대상: `LayerNorm` 파라미터 및 `GraphINN` flow 블록
- 로컬 자산: 선택한 backbone의 pretrained 가중치
- 구현 특이사항: `LayerNormFeatureExtractor`를 통해 계층별 feature를 정규화한 뒤 U-Net 형태의 multi-scale flow graph(`IRevNetUpsampling`, `Split`, `Concat`, `AllInOneBlock`)로 처리한다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `backbone` | `str` | `resnet18` | Config selector: `resnet18`, `wide_resnet50_2`, `mcait` | feature 추출에 사용할 고정 backbone |
| `weights_path` | `str \| None` | `${paths.backbone_root}/resnet18-f37072fd.pth` | 선택한 backbone과 일치하는 로컬 파일 경로 | backbone에 적용할 pretrained 가중치 경로 |
| `input_size` | `list[int]` | `[256, 256]` | `[H, W]` 형태의 이미지 해상도 | 입력 이미지 크기 |
| `flow_steps` | `int` | `4` | 양의 정수 | 각 스케일별 coupling block 단계 수 |
| `affine_clamp` | `float` | `2.0` | 양수 | affine coupling layer의 clamping 값 |
| `affine_subnet_channels_ratio` | `float` | `1.0` | 양수 | affine coupling subnet의 중간 채널 비율 |
| `permute_soft` | `bool` | `false` | `true`/`false` | soft permutation 사용 여부 |

### 3.13 CS-Flow

CS-Flow(Cross-Scale Flow)는 EfficientNet-B5에서 추출한 3개 스케일의 feature map을 상호 연결된 cross-scale coupling convolution layer로 변환하여 다중 스케일 결합 확률 밀도를 학습한다. 학습 시 각 스케일의 잠재 변수와 log-Jacobian determinant를 결합한 손실을 사용하고, 추론 시 다중 스케일 anomaly map을 합성하여 이상 점수와 히트맵을 생성한다.

- 모델 factory: `csflow_anomaly` (별칭: `csflow`)
- Adapter: `csflow`
- Config: `configs/anomaly/models/csflow.yaml`
- 기본 backbone: `efficientnet_b5` (`features.6.8` 레이어 사용)
- 학습 대상: `CrossScaleFlow` (`ParallelGlowCouplingLayer`, `CrossConvolutions`, `ParallelPermute`) 파라미터
- 로컬 자산: `${paths.backbone_root}/efficientnet_b5_lukemelas-1a07897c.pth`
- 구현 특이사항: FX-traced `TimmFeatureExtractor`를 사용해 EfficientNet-B5에서 3개 해상도(원 해상도, 1/2, 1/4)로 다운샘플링된 feature를 추출하며, backbone은 freeze된다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `weights_path` | `str \| None` | `${paths.backbone_root}/efficientnet_b5_lukemelas-1a07897c.pth` | 로컬 파일 경로 | EfficientNet-B5 pretrained 가중치 경로 |
| `input_size` | `list[int]` | `[256, 256]` | `[H, W]` 형태의 이미지 해상도 | 입력 이미지 크기 |
| `cross_conv_hidden_channels` | `int` | `1024` | 양의 정수 (예: 64, 512, 1024) | cross-scale convolution의 은닉 채널 수 |
| `n_coupling_blocks` | `int` | `4` | 양의 정수 | cross-scale coupling block 단계 수 |
| `clamp` | `float` | `3.0` | 양수 | coupling layer의 clamping 값 |
| `num_channels` | `int` | `3` | `3` | 입력 이미지 채널 수 |

### 3.14 SuperSimpleNet

SuperSimpleNet은 사전 학습된 CNN feature를 2배 업스케일 및 AvgPool2d로 인접 패치를 집약한 뒤, 1x1 projection adapter와 train-time feature-level 합성 이상 생성기(AnomalyGenerator)를 결합하여 정상 feature와 이상 feature를 판별(Discriminative)하도록 학습하는 모델이다. Segmentation 헤드(1x1 Conv + LeakyReLU)와 Classification 헤드(Conv + pooling + FC)를 통해 anomaly map과 anomaly score를 예측하며 빠른 추론 속도를 제공한다.

- 모델 factory: `supersimplenet_anomaly` (별칭: `supersimplenet`)
- Adapter: `supersimplenet`
- Config: `configs/anomaly/models/supersimplenet.yaml`
- 기본 backbone: `wide_resnet50_2` (`layer2`, `layer3` 레이어 사용)
- Selector: `--model.backbone wide_resnet50_2|resnet18|resnet50`
- 학습 대상: `FeatureAdapter` (`projection`), `SegmentationDetectionModule` (`seg_head`, `cls_conv`, `cls_fc`) 파라미터
- 로컬 자산: `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` (또는 선택한 backbone 가중치)
- 구현 특이사항: 학습 중 `AnomalyGenerator`가 Perlin noise 패턴과 Gaussian 노이즈를 결합해 feature 공간에서 합성 이상 맵과 라벨을 생성하고, `SSNLoss` (Focal Loss + Truncated L1 Loss)로 학습한다. Backbone은 freeze된다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `weights_path` | `str \| None` | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` | 로컬 파일 경로 | Backbone pretrained 가중치 경로 |
| `backbone` | `str` | `"wide_resnet50_2"` | `wide_resnet50_2`, `resnet18`, `resnet50` | Feature extractor 백본 이름 |
| `layers` | `list[str]` | `["layer2", "layer3"]` | 백본 레이어 리스트 | 추출에 사용할 중간 레이어 이름들 |
| `perlin_threshold` | `float` | `0.2` | `0.0` ~ `1.0` | Anomaly generation용 Perlin 노이즈 binarization threshold |
| `stop_grad` | `bool` | `true` | `true`/`false` | classification 헤드에서 segmentation 헤드로의 gradient 차단 여부 |
| `adapt_cls_features` | `bool` | `false` | `true`/`false` | classification 헤드 입력으로 adapted feature를 사용할지 여부 |

### 3.15 GANomaly

GANomaly는 적대적 오토인코더 구조(Generator: Encoder1-Decoder-Encoder2)와 Discriminator(Encoder-Classifier)를 결합하여 정상 이미지를 잠재 공간으로 압축·재구성하고, 첫 번째 잠재 벡터 $z$와 재구성 이미지의 잠재 벡터 $\hat{z}$ 사이의 거리를 이미지 단위 이상 점수로 산출하는 모델이다 (Pixel anomaly map 미생성).

- 모델 factory: `ganomaly_anomaly` (별칭: `ganomaly`)
- Adapter: `ganomaly`
- Config: `configs/anomaly/models/ganomaly.yaml`
- 기본 크기: 자체 합성곱 인코더/디코더
- 학습 대상: Generator 및 Discriminator 파라미터 전체 (From scratch 학습)
- 로컬 자산: 없음
- 구현 특이사항: `GanomalyAdapter`가 Generator 손실(Adversarial + Contextual + Latent Error)과 Discriminator 손실(BCE)을 전용 Dual Adam 옵티마이저로 번갈아 최적화한다.
- **성능 기대치 주의**: MVTec에서 image AUROC가 0.5 미만으로 나오는 것이 **정상이며 포팅 결함이 아니다.** anomalib 공식 벤치마크(`ganomaly/README.md`) 기준 MVTec 평균 image AUROC는 **0.421**이고 15개 카테고리 중 12개가 0.5 미만이다(bottle 0.251, capsule 0.682). 이미지 전체를 100차원 latent로 압축해 그 재구성 오차만으로 점수를 내는 구조라, 국소 결함이 대부분인 MVTec에는 원리적으로 맞지 않는다(원래 MNIST/CIFAR의 클래스 단위 이상 탐지용). 실사용 후보로는 권장하지 않는다 — 상세는 `docs/dev/v0.2/reports/UPSTREAM-INVENTORY.md` §23.1.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `input_size` | `list[int]` | `[256, 256]` | `[H, W]` 형태의 이미지 해상도 | 입력 이미지 크기 |
| `n_features` | `int` | `64` | 양의 정수 | 초기 합성곱 필터 채널 수 |
| `latent_vec_size` | `int` | `100` | 양의 정수 | 잠재 공간(Bottleneck) 벡터 차원 |
| `extra_layers` | `int` | `0` | `0` 이상의 정수 | 인코더/디코더에 추가할 중간 합성곱 계층 수 |
| `add_final_conv_layer` | `bool` | `true` | `true`/`false` | 최종 출력 합성곱 계층 추가 여부 |

### 3.16 DRAEM

DRAEM(Discriminatively Trained Reconstruction Embedding)은 Reconstructive SubNetwork(AutoEncoder)와 Discriminative SubNetwork(U-Net 스타일 판별기)를 결합하여, DTD 텍스처 데이터셋과 Perlin 노이즈로 생성된 인공 결함(Simulated anomaly)을 정상 이미지로 복원하고 동시에 결함 영역을 픽셀 단위로 분할하도록 지도학습하는 모델이다.

- 모델 factory: `draem_anomaly` (별칭: `draem`)
- Adapter: `draem`
- Config: `configs/anomaly/models/draem.yaml`
- 기본 크기: 자체 Reconstructive/Discriminative 서브네트워크
- 학습 대상: Reconstructive SubNetwork 및 Discriminative SubNetwork 파라미터 전체
- 로컬 자산: `${paths.dataset_root}/dtd` (DTD 텍스처 데이터셋)
- 구현 특이사항: `DraemAdapter`가 학습 중 `PerlinAnomalyGenerator`로 결함 이미지를 생성하고, `DraemLoss` (L2 Reconstruction + SSIM + Focal Segmentation Loss)로 최적화한다. SSPCAB 모듈을 활성화할 경우 bottleneck attention 손실을 추가할 수 있다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `sspcab` | `bool` | `false` | `true`/`false` | SSPCAB (Self-Supervised Predictive Convolutional Attention Block) 활성화 여부 |

### 3.17 DSR

DSR(Dual Subspace Re-Projection Network)은 사전학습된 VQ-VAE 이산 잠재 모델(Discrete Latent Model)의 양자화 코드북을 기반으로, Subspace Restriction Module을 통해 결함 영역을 정상 부분공간으로 재투영하고 Anomaly Detection Module과 Upsampling Module을 거쳐 픽셀 단위 이상 맵을 생성하는 모델이다.

- 모델 factory: `dsr_anomaly` (별칭: `dsr`)
- Adapter: `dsr`
- Config: `configs/anomaly/models/dsr.yaml`
- 기본 크기: Discrete Latent Model + Subspace Restriction + Upsampling Module
- 학습 대상: Image Reconstruction, Subspace Restriction Hi/Lo, Anomaly Detection Module (Phase 1) 및 Upsampling Module (Phase 2)
- 로컬 자산: `${paths.backbone_root}/vq_model_pretrained_128_4096.pckl` (사전학습 VQ-VAE 코드북)
- 구현 특이사항: `DsrAdapter`가 2단계 학습(Phase 1: 양자화 결함 재투영 및 탐지 모듈 학습, Phase 2: Perlin smudge 기반 업샘플링 모듈 학습)을 자동으로 전환하며, 이산 VQ-VAE 백본은 항상 freeze 상태를 유지한다. Phase 경계는 `int(train.epochs × upsampling_train_ratio)`로 정해지고, 전체 epoch 수는 engine이 `adapter.total_epochs`로 공급한다.
- **정규화 금지**: upstream `Dsr.on_train_start`가 `Normalize`를 발견하면 `ValueError`로 즉시 실패한다(사전학습 VQ-VAE 코드북이 [0,1] 이미지로 학습됨). `dsr.yaml`이 `data.transform.{train,eval}.params.normalize: false`로 이를 강제한다 — 제거하면 안 된다.
- **최소 epoch 수 주의**: `--epochs 1`로 실행하면 `int(1 × 0.7) = 0`이 되어 Phase 1이 아예 실행되지 않는다(upstream도 동일한 축퇴 동작). 스모크 검증도 반드시 Phase 경계를 넘는 epoch 수로 해야 한다. 논문 수준 성능(image AUROC ~0.98)에는 100 epoch 이상이 필요하며, 10 epoch에서는 0.667 수준이다 — 상세는 `docs/dev/v0.2/reports/UPSTREAM-INVENTORY.md` §23.2.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `weights_path` | `str \| None` | `${paths.backbone_root}/vq_model_pretrained_128_4096.pckl` | 로컬 파일 경로 | 사전학습된 VQ-VAE 이산 코드북 가중치 경로 |
| `latent_anomaly_strength` | `float` | `0.2` | `0.0` ~ `1.0` | 잠재 공간 인공 결함 강도 |
| `embedding_dim` | `int` | `128` | 양의 정수 | 코드북 임베딩 차원 |
| `num_embeddings` | `int` | `4096` | 양의 정수 | 코드북 임베딩 벡터 수 |
| `num_hiddens` | `int` | `128` | 양의 정수 | 은닉 채널 수 |
| `num_residual_layers` | `int` | `2` | 양의 정수 | Residual 블록 수 |
| `num_residual_hiddens` | `int` | `64` | 양의 정수 | Residual 은닉 채널 수 |

### 3.18 UniNet

UniNet은 고정된 source teacher와 학습 가능한 target teacher 이중 구조에서 추출한 feature를 attention bottleneck으로 결합하고, DFS(Domain-related Feature Selection)로 선별한 뒤 student(ResNet decoder)가 두 teacher의 feature를 동시에 근사하도록 학습하는 모델이다.

- 모델 factory: `uninet_anomaly` (별칭: `uninet`)
- Adapter: `uninet`
- Config: `configs/anomaly/models/uninet.yaml`
- 기본 backbone: `wide_resnet50_2` (student/teacher 공통, selector로 `resnet18`/`resnet50` 전환 가능)
- 학습 대상: student(ResNet decoder), bottleneck, DFS, target teacher (source teacher는 영구 freeze)
- 로컬 자산: `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` (teacher/student 공통 초기 가중치)
- 구현 특이사항: `UniNetAdapter.configure_optimizers`가 upstream의 split-LR AdamW(student+bottleneck+dfs @ 5e-3, target_teacher @ 1e-6)를 그대로 재현한다. `source_teacher`는 인스턴스 수준 `train` 메서드 오버라이드로 공통 engine의 매 epoch `model.train()` 재귀 호출에도 eval 고정을 유지한다. MVTec train split이 빈 target(`{}`)만 제공하므로 all-zero-label로 대체하는데, all-normal 배치에서는 실제 all-zero per-pixel mask와 수학적으로 동치다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `weights_path` | `str \| None` | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` | 로컬 파일 경로 | teacher/student 초기 가중치 경로 |
| `student_backbone` | `str` | `wide_resnet50_2` | `resnet18`, `resnet50`, `wide_resnet50_2` | student decoder 백본 |
| `teacher_backbone` | `str` | `wide_resnet50_2` | `resnet18`, `resnet50`, `wide_resnet50_2` | teacher 백본 |
| `temperature` | `float` | `0.1` | 양의 실수 | contrastive loss 온도 |

### 3.19 Dinomaly

Dinomaly는 frozen DINOv2 ViT encoder에서 추출한 다중 계층 feature를 bottleneck MLP로 압축한 뒤 ViT decoder로 재구성하고, encoder-decoder 간 코사인 유사도를 anomaly signal로 사용하는 모델이다.

- 모델 factory: `dinomaly_anomaly` (별칭: `dinomaly`)
- Adapter: `dinomaly`
- Config: `configs/anomaly/models/dinomaly.yaml`
- 기본 encoder: `dinov2reg_vit_base_14` (selector로 small/large 전환 가능)
- 학습 대상: bottleneck, decoder (encoder는 영구 freeze)
- 로컬 자산: `${paths.backbone_root}/dinov2_vit*_[reg4_]pretrain.pth` (DINOv2 pretrained 가중치)
- 구현 특이사항: upstream의 per-step `WarmCosineScheduler` + `StableAdamW`는 공통 engine이 `scheduler.step()`을 epoch당 1회만 호출하는 구조와 맞지 않아, `DinomalyAdapter`가 private optimizer/scheduler를 소유하고 매 배치 zero_grad/backward/step을 직접 수행한 뒤 zero dummy loss를 engine에 반환한다(`CflowAdapter`와 동일 패턴, private step 후 `.grad`를 반드시 다시 비워야 engine 소유 optimizer가 stale gradient로 이중 업데이트하지 않는다). `DinoV2Loader`가 캐시 디렉터리를 하드코딩하므로 `build_dinomaly`가 생성 구간에서만 `__init__`을 몽키패치해 로컬 `paths.backbone_root`를 가리키게 전환하고, `components/dinov2/local_preflight.py`로 실제 요청될 파일 존재를 생성 전에 강제 검증한다. `data.image_size=[392,392]`는 upstream의 Resize(448)+CenterCrop(392) 근사치다(현재 transform에 crop 단계 없음).

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `weights_path` | `str \| None` | `${paths.backbone_root}/dinov2_vitb14_reg4_pretrain.pth` | 로컬 파일 경로 (디렉터리만 실제 사용) | DINOv2 pretrained 가중치가 있는 디렉터리를 가리키는 파일 |
| `encoder_name` | `str` | `dinov2reg_vit_base_14` | `dinov2reg_vit_{small,base,large}_14` | DINOv2 encoder 아키텍처 |
| `bottleneck_dropout` | `float` | `0.2` | `0.0` ~ `1.0` | bottleneck MLP dropout |
| `decoder_depth` | `int` | `8` | 양의 정수 (2 이상) | ViT decoder 블록 수 |

### 3.20 AnomalyDINO

AnomalyDINO는 frozen DINOv2 ViT의 patch feature를 memory bank에 저장하고 최근접 이웃 코사인 거리를 anomaly signal로 사용하는, PatchCore와 동일한 메커니즘의 no-gradient 모델이다.

- 모델 factory: `anomaly_dino_anomaly` (별칭: `anomaly_dino`)
- Adapter: `anomaly_dino`
- Config: `configs/anomaly/models/anomaly_dino.yaml`
- 기본 encoder: `dinov2_vit_small_14`
- 학습 대상: 없음 (전체 freeze, memory bank만 구축)
- 로컬 자산: `${paths.backbone_root}/dinov2_vit*_pretrain.pth` (DINOv2 pretrained 가중치)
- 구현 특이사항: PatchCore와 동일한 `KCenterGreedy`(coreset subsampling)와 `AnomalyMapGenerator`를 재사용한다. `AnomalyDinoAdapter.on_validation_start`가 `model.fit()`으로 memory bank를 1회 확정한다(PatchCore의 `subsample_embedding`과 동일 시점). DINOv2 로더의 다운로드 폴백 방지는 Dinomaly와 동일하게 팩토리 몽키패치 + `local_preflight`로 처리한다. fit-only 구조라 1 epoch로 학습이 완결된다.

#### 모델 파라미터

| 파라미터 | 타입 | 기본값 | 지원값 | 설명 |
|---|---|---|---|---|
| `weights_path` | `str \| None` | `${paths.backbone_root}/dinov2_vits14_pretrain.pth` | 로컬 파일 경로 (디렉터리만 실제 사용) | DINOv2 pretrained 가중치가 있는 디렉터리를 가리키는 파일 |
| `encoder_name` | `str` | `dinov2_vit_small_14` | `dinov2_vit_{small,base,large}_14` | DINOv2 encoder 아키텍처 |
| `num_neighbours` | `int` | `1` | 양의 정수 | kNN 탐색 이웃 수 |
| `masking` | `bool` | `False` | `True`/`False` | PCA 기반 배경 마스킹 사용 여부 |
| `coreset_subsampling` | `bool` | `False` | `True`/`False` | greedy coreset 축소 사용 여부 |
| `sampling_ratio` | `float` | `0.1` | `0.0` ~ `1.0` | coreset 샘플링 비율 |

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
| CFLOW | fiber 단위 decoder loss 계산 및 **adapter 소유 private optimizer**로 fiber마다 즉시 step (§3.10) |
| FRE | CNN feature와 Tied AutoEncoder 재구성 간 MSE 손실 계산 |
| U-Flow | 다중 스케일 잠재 변수 및 log-Jacobian 결합 손실 계산 |
| CS-Flow | 다중 스케일 cross-scale flow 잠재 변수 및 log-Jacobian 손실 계산 |
| SuperSimpleNet | Feature-level 합성 이상 생성 및 Focal + Truncated L1 결합 손실(`SSNLoss`) 계산 |
| GANomaly | Generator/Discriminator 이원 적대적 손실 계산 및 전용 Dual Adam 옵티마이저 스텝 |
| DRAEM | DTD/Perlin 합성 인공 결함 생성 및 Reconstructive-Discriminative 복합 손실 계산 |
| DSR | 2단계 학습 모듈 전환(Reconstruction / Upsampling) 및 다중 옵티마이저 스텝 |

CFLOW, GANomaly, DSR은 이 표의 다른 모델과 달리 공통 engine의 단일 optimizer에 의존하지 않고, 각각의 adapter가 전용 private optimizer들을 직접 소유·스케줄링하여 다단계/다중 최적화를 완벽히 수행한다.

### 4.3 Offline 실행

모델 생성 과정에서 pretrained 가중치를 자동으로 내려받지 않는다. 가중치와 보조 데이터는 `configs/local.yaml` 또는 환경변수로 지정한 root 아래에 사용자가 준비해야 한다. 필요한 파일이나 폴더가 없으면 실행을 계속하지 않고 오류로 종료한다.

### 4.4 AUROC 지표

`image_auroc`·`pixel_auroc`는 `src/tasks/anomaly/metrics/rank_auroc.py`의 `RankAUROC`를 쓴다. torchmetrics의 `BinaryAUROC`는 `preds`가 [0,1]을 벗어나면 자동으로 sigmoid를 적용하는데, float32 sigmoid는 대략 `x > 17`부터 정확히 `1.0`으로 포화한다. 포화하면 모든 값이 동점이 되어 AUROC가 정확히 0.5로 붕괴하고, 부분 포화는 오답을 동점으로 바꿔 AUROC를 부풀린다.

anomaly map과 anomaly score는 모델마다 스케일이 크게 다르므로(CS-Flow의 맵은 3개 스케일 `mean(z²)`의 곱이라 수백~수만 규모) 이 제약은 실제로 문제가 된다. `RankAUROC`는 입력을 변환 없이 받아 Mann-Whitney U 통계량을 동점 보정과 함께 계산한다. 포화가 없던 구간에서는 `BinaryAUROC`와 값이 동일하다.

**모델을 추가할 때**: anomaly map이나 score의 값 범위를 확인할 필요는 없다. 다만 어떤 지표가 여러 epoch에 걸쳐 정확히 `0.500`으로 고정된다면 값 포화나 상수 출력을 먼저 의심한다.

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
| CFLOW | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` |
| FRE | `${paths.backbone_root}/resnet50-0676ba61.pth` |
| U-Flow | `${paths.backbone_root}/resnet18-f37072fd.pth` |
| CS-Flow | `${paths.backbone_root}/efficientnet_b5_lukemelas-1a07897c.pth` |
| SuperSimpleNet | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` |
| GANomaly | 없음 (from scratch) |
| DRAEM | `${paths.dataset_root}/dtd` |
| DSR | `${paths.backbone_root}/vq_model_pretrained_128_4096.pckl` |
| UniNet | `${paths.backbone_root}/wide_resnet50_2-95faca4d.pth` |
| Dinomaly | `${paths.backbone_root}/dinov2_vitb14_reg4_pretrain.pth` (디렉터리 전체 `dinov2_vit*_[reg4_]pretrain.pth` 필요) |
| AnomalyDINO | `${paths.backbone_root}/dinov2_vits14_pretrain.pth` (디렉터리 전체 `dinov2_vit*_pretrain.pth` 필요) |

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
| CFLOW | `timm`, `FrEIA` |
| FRE | `timm` |
| U-Flow | `timm`, `FrEIA`, `scipy`, `omegaconf` |
| CS-Flow | `torchvision`, `FrEIA`, `numpy` |
| SuperSimpleNet | `timm`, `torchvision`(sigmoid_focal_loss) |
| GANomaly | `torch`, `torchvision` |
| DRAEM | `kornia`, `torchvision` |
| DSR | `kornia`, `torchvision` |

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
--model configs/anomaly/models/cflow.yaml
--model configs/anomaly/models/fre.yaml
--model configs/anomaly/models/uflow.yaml
--model configs/anomaly/models/csflow.yaml
--model configs/anomaly/models/supersimplenet.yaml
--model configs/anomaly/models/ganomaly.yaml
--model configs/anomaly/models/draem.yaml
--model configs/anomaly/models/dsr.yaml
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
