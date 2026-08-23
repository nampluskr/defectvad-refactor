# PLAN — anomalib 모델 포팅 우선순위

문서 상태: Approved Plan (사용자 승인)
상위 문서: `BRIEF.md`
관련 문서: `docs/guides/anomaly-models.md` (모델 구현 현황), `.claude/skills/add-anomalib-model/SKILL.md` (`/add-anomalib-model` 스킬 포팅 절차), `docs/dev/v0.1/reports/MODEL-ADD.md` (포팅 절차 원문)
작성일: 2026-08-23
최종 갱신: 2026-08-23

## 1. 배경

레거시 `defectvad@14879ea2`에 구현된 anomaly detection 모델 **20개 전체**를 이 프로젝트(`cv_boilerplate` 기반 v0.2 refactor)에 포팅하는 것이 목표다. 현재 10개 모델이 구현 완료되었고, 남은 10개 모델을 이 문서의 우선순위에 따라 포팅한다.

anomalib 핀 `091ca6a`(v2.3.0)의 `src/anomalib/models/image/` 디렉터리에 20개 모델 중 WinCLIP·VLM-AD를 제외한 **18개가 모두 존재**함을 확인했다. 레거시 defectvad에 없는 WinCLIP·VLM-AD는 이 PLAN의 범위 밖이다.

## 2. 우선순위 판단 기준

1. **오프라인 원칙 준수** -- `BRIEF.md` 원칙3(로컬 자산만 사용)을 위반하지 않는가. 인터넷 접근이나 외부 API가 필요한 모델은 후순위로 미룬다.
2. **패러다임 다양성** -- `docs/guides/anomaly-models.md` §2의 분류 체계에서 아직 비어 있는 패러다임(Reconstruction, Density Estimation 등)을 채우는가.
3. **포팅 비용** -- 기존 컴포넌트(`components/feature_extractors`, `FrEIA` 등)를 재사용해 낮은 비용으로 포팅 가능한가, 아니면 새로운 학습 구조(2단계 학습, adversarial loss, backbone fine-tuning 등)가 필요한가.
4. **레거시 대등성** -- 레거시 defectvad에 구현된 모든 모델을 빠짐없이 포팅한다.

## 3. 구현 완료 (참고, 재작업 대상 아님)

| 패러다임 | 모델 |
|---|---|
| Teacher-Student / Knowledge Distillation | STFPM, EfficientAD, Reverse Distillation |
| Normalizing Flow | FastFlow |
| Feature Embedding / Memory Bank | PatchCore, PaDiM |
| Density Estimation | DFM, DFKDE |
| Feature Embedding / Memory Bank (gradient 학습) | CFA |
| Normalizing Flow (fiber 단위 학습) | CFLOW |
| Autoencoder / Reconstruction | FRE |
| Normalizing Flow (multi-scale) | U-Flow |
| Normalizing Flow (cross-scale) | CS-Flow |
| Discriminative (synthetic anomaly) | SuperSimpleNet |
| Reconstruction (adversarial) | GANomaly |
| Reconstruction (discriminative) | DRAEM |
| Reconstruction (discrete latent) | DSR |

DFM/DFKDE/CFA는 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했고, 반대 벤더(Codex CLI) 적대적 검증을 1회 거쳤다(`docs/dev/v0.2/reviews/A1.md`). CFA는 사용자가 실제로 `scripts/train.py`(bottle)를 실행해 학습이 정상 완주됨을 확인했다 -- 이 과정에서 적대적 검토가 잡아내지 못한 결함 2건(YAML `1e-5` 파싱 함정, `CfaLoss.radius`의 non-leaf 텐서 재사용으로 인한 2스텝째 backward 실패)이 드러나 수정했다(`docs/dev/v0.2/reports/UPSTREAM-INVENTORY.md` §12.4). 이후 사용자가 DFM/DFKDE/CFA 세 모델 모두 train/evaluate/predict를 직접 실행해 정상 동작을 확인했다(2026-08-23) -- 구체적인 image/pixel AUROC 수치는 별도로 기록되지 않았다. 3개 카테고리(bottle/carpet/capsule) 기준 정식 성능 비교는 여전히 사용자 실행 대기 상태다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.7~§3.9를 참조한다.

CFA는 착수 전 우려했던 "backbone fine-tune으로 SSOT 전제가 흔들리는 사례"가 실제로는 발생하지 않았다: 모델 원본은 backbone을 얼린 채로 두지 않지만(anomalib 자체가 `model.parameters()` 전체를 optimizer에 넘김), forward가 항상 `torch.no_grad()`로 backbone을 실행하므로 그레이디언트가 도달하지 않는다. 이 프로젝트는 factory에서 backbone을 `requires_grad=False`로 명시적으로 고정해 다른 모델과 동일한 "backbone 고정" 관례를 유지했다 -- 결과는 anomalib과 동일하다.

CFLOW는 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했고, 반대 벤더(Codex CLI) 적대적 검증을 2회 거쳤다(`docs/dev/v0.2/reviews/A2.md`). Upstream이 `automatic_optimization=False`로 이미지 배치당 fiber(feature map 조각) 단위로 최대 168회 별도 Adam step을 수행하는 구조라, 이 프로젝트의 "배치당 1회 zero_grad+backward+step" 공통 engine 계약과 정면으로 부딪혔다 -- 1차 검토에서 이를 "fiber loss 합산 후 1회 step"으로 근사했다가 Major로 지적받아, `CflowAdapter`가 decoder 전용 private `torch.optim.Adam`을 직접 소유하고 fiber마다 upstream과 동일하게 즉시 step하도록 재작성했다(engine 소유 optimizer는 항상 0-gradient만 받는 더미 loss로 무력화). bottle 카테고리 1-epoch 스모크에서 이 수정으로 test image_auroc 0.908->1.000, pixel_auroc 0.927->0.987로 개선을 실측했다. 남은 미해결 사항은 `--resume` 시 이 private optimizer의 Adam 모멘텀이 보존되지 않는 것(공통 engine에 adapter-state checkpoint 훅이 없어 발생, NFR-005상 이번 세션 범위 밖)과, `runtime.amp`/`train.grad_clip`이 CFLOW의 실제 업데이트를 제어하지 못하는 것(기본 config에서는 무해) 두 가지이며, 둘 다 `CflowAdapter` docstring과 A2.md에 명시했다. 상세 구현 개요는 `docs/guides/anomaly-models.md`를 포팅 절차에 따라 추가 갱신할 때 반영한다.

FRE는 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했다. Tied AutoEncoder 기반으로 CNN backbone feature의 재구성 MSE 손실을 Adam optimizer로 학습하는 구조이며, train/evaluate/predict 3종 스모크 검증을 통과했다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.11을 참조한다.

U-Flow는 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했다. `LayerNormFeatureExtractor` 기반의 다중 스케일 feature 추출 및 U자형 FrEIA normalizing flow graph를 결합하여 log-likelihood + log-Jacobian 손실로 학습하며, train/evaluate/predict 3종 스모크 검증을 통과했다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.12를 참조한다.

CS-Flow는 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했다. EfficientNet-B5에서 추출한 3개 스케일 feature의 상호 결합 cross-scale flow를 Adam optimizer로 학습하며, train/evaluate/predict 3종 스모크 검증을 통과했다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.13을 참조한다.

SuperSimpleNet은 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했다. Feature-level Perlin 노이즈 합성 기반 anomaly generator와 segmentation-detection 모듈을 결합하여 Focal + Truncated L1 손실로 학습하며, train/evaluate/predict 3종 스모크 검증을 통과했다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.14를 참조한다.

GANomaly는 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했다. Generator(Encoder-Decoder-Encoder)와 Discriminator의 이원 적대적 손실(BCE + L1 + SmoothL1 + MSE)을 Dual Adam optimizer로 학습하며, train/evaluate/predict 3종 스모크 검증을 통과했다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.15를 참조한다.

DRAEM은 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했다. Reconstructive SubNetwork 및 Discriminative SubNetwork에 DTD/Perlin 합성 인공 결함을 주입하여 L2 + SSIM + Focal 복합 손실로 학습하며, train/evaluate/predict 3종 스모크 검증을 통과했다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.16을 참조한다.

DSR은 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했다. Pretrained VQ-VAE 이산 코드북을 기반으로 Subspace Restriction Module과 Anomaly Detection Module을 거쳐 2단계(Reconstruction + Upsampling) 다중 옵티마이저로 학습하며, train/evaluate/predict 3종 스모크 검증을 통과했다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.17을 참조한다.

## 4. 우선순위 표 (미구현 3개 모델)

### 4.1 Tier 1 -- 기존 컴포넌트 재사용, anomalib SSOT 직접 적용

Tier 1 모델 4종(FRE, U-Flow, CS-Flow, SuperSimpleNet) 전량 포팅 및 적대적 교차 검증(A3) 승인 완료.

### 4.2 Tier 2 -- 새로운 학습 구조가 필요하거나 adapter 복잡도 상승

Tier 2 모델 3종(GANomaly, DRAEM, DSR) 전량 포팅 및 적대적 교차 검증(A4) 승인 완료.

### 4.3 Tier 3 -- Foundation Model backbone, 추가 로컬 자산 확보 필요

DINOv2 등 대형 pretrained backbone을 사용하며, 해당 가중치의 로컬 확보와 오프라인 로딩 방식 검증이 선행되어야 한다.

| 순위 | 모델 | anomalib 경로 | 패러다임 | 학습 방식 | 근거 | 리스크/확인 필요 사항 |
|---|---|---|---|---|---|---|
| 8 | UniNet | `image/uninet` | Teacher-Student (multi-branch) | gradient training | attention bottleneck, DFS 등 고유 컴포넌트 다수 (`attention_bottleneck.py`, `dfs.py`) | anomalib에 존재하나 구조 복잡도 높음. backbone 가중치 로컬화 확인 |
| 9 | DinoMaly | `image/dinomaly` | Reconstruction (ViT 기반) | gradient training | DINOv2 backbone 활용 anomaly detection | DINOv2 pretrained 가중치(`vit_base_patch14_dinov2.lvd142m` 등)의 로컬 확보 필요. `torch.hub` 자동 다운로드 차단 후 로컬 주입 방식 확인 |
| 10 | AnomalyDINO | `image/anomaly_dino` | Feature Embedding (ViT 기반) | no-gradient 또는 lightweight | DINO/DINOv2 backbone feature 활용 | DINOv2 가중치 로컬 확보 필요. 학습 없는 few-shot/zero-shot 방식일 경우 기존 train-evaluate-predict 파이프라인과의 적합성 확인 필요 |

### 4.4 범위 밖 (레거시 defectvad에 없음)

| 모델 | 사유 |
|---|---|
| WinCLIP | 레거시 defectvad에 없음. Vision-Language 모델, CLIP 가중치 필요, 오프라인 원칙 저촉 |
| VLM-AD | 레거시 defectvad에 없음. VLM 기반, 외부 API 의존 가능성 |

## 5. 완료 검증 조건

### 5.1 모델별 완료 조건 (모든 모델에 공통 적용)

모델 1개의 포팅이 "완료"로 인정되려면 아래 7개 조건을 **모두** 충족해야 한다.

| # | 조건 | 검증 방법 | 근거 |
|---|---|---|---|
| V-01 | anomalib 원본 파일이 import 경로 외에 무수정 | `diff -u` 원본 대조, sha256 기록 | CON-001 (원칙 1) |
| V-02 | Lightning import/의존/호출 경로 없음 | `grep -rn "lightning" src/tasks/anomaly/models/<model>/` (docstring 제외) | CON-002 (원칙 2) |
| V-03 | 자동 다운로드 경로 없음, 로컬 자산만 사용 | `OfflineViolationError` 미발생 확인, `check-assets` 통과 | CON-003, CON-004 (원칙 3) |
| V-04 | `MODELS`/`ADAPTERS` registry 등록 완료 | `adapters/__init__.py` import 존재, `RegistryError` 미발생 | `/add-anomalib-model` §3.5 |
| V-05 | 단일 카테고리 스모크 통과 (train/evaluate/predict 3종) | `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py` 각 1회 이상 정상 완료 | `/add-anomalib-model` §3.7 |
| V-06 | 공통 engine 오염 없음 | `core/` 디렉터리에 모델명/task명 분기 없음 확인 | NFR-005 |
| V-07 | model config 작성 완료 | `configs/anomaly/models/<model>.yaml` 존재, `${paths.*}` placeholder 사용 | `/add-anomalib-model` §3.6 |

### 5.2 추가 검증 (권장, 포팅 완료 후 순차 수행)

| # | 조건 | 검증 방법 | 비고 |
|---|---|---|---|
| V-08 | 반대 벤더 적대적 검증 통과 | `docs/dev/v0.2/reviews/A{n}.md` 작성, Critical 0건 | 최대 3회 |
| V-09 | `UPSTREAM-INVENTORY.md` 갱신 | 파일 인벤토리, 라이선스, 허용 변경 기록 | v0.2 문서 |
| V-10 | `docs/guides/anomaly-models.md` 갱신 | 전체 구현 현황표, 분류 체계, 모델별 구현 개요 추가 | §7 문서 갱신 기준 |
| V-11 | 사용자 실행 확인 (3개 카테고리) | bottle/carpet/capsule 기준 train/evaluate 실행, image_auroc 수치 기록 | 사용자 실행 |

### 5.3 전체 포팅 완료 조건

레거시 defectvad의 모든 모델 포팅이 "완료"로 인정되려면:

1. 20개 모델 전부 §5.1의 V-01~V-07을 충족한다.
2. 20개 모델 전부 §3의 "구현 완료" 표에 등재된다.
3. `docs/guides/anomaly-models.md`의 전체 구현 현황표에 20개 모델이 모두 "구현됨"으로 기록된다.
4. 모든 모델에 대해 적대적 검증(V-08)을 1회 이상 수행한다.

## 6. 미정 사항

- CFLOW의 `--resume` 시 decoder Adam 모멘텀 미보존 문제 -- 공통 engine에 adapter-state checkpoint 훅(`src/core/checkpoint.py`의 `adapter_state` 매개변수는 이미 존재하나 `src/core/engine.py`가 채우지 않는 미완성 확장점)을 추가하는 별도 과제로 이월. CS-Flow 등 이후 fiber/2단계 학습 모델에서도 같은 문제가 반복될 가능성이 높다.
- DRAEM 착수 전 원칙3(오프라인) 저촉 여부 재검토 필요 -- synthetic anomaly 생성용 texture 데이터셋 로컬화 방법 확정 (§4.2 표의 "리스크" 열 참조).
- DinoMaly/AnomalyDINO 착수 전 DINOv2 pretrained 가중치의 로컬 확보 방법 확정 필요.
- DFM/DFKDE/CFA는 세 스크립트(train/evaluate/predict) 실행 자체는 사용자가 확인했다(2026-08-23). 3개 카테고리(bottle/carpet/capsule) 기준 정식 성능 비교와 수치 기록은 아직 없다 -- 필요 시 결과에 따라 config 하이퍼파라미터(특히 CFA `train.epochs`, DFM `score_type`)를 조정할 수 있다.
- PatchCore/PaDiM/DFM/DFKDE의 `runtime.amp: true` 비호환 가능성, `weights_path=None` 시 silent random-init -- 적대적 검토(A1)에서 지적됐으나 9개 모델에 걸친 기존 설계라 이번 세션에서는 수정하지 않았다. 별도 과제로 core 변경 필요 (`UPSTREAM-INVENTORY.md` §14).
- 각 모델 착수 시 `/add-anomalib-model` 스킬(`.claude/skills/add-anomalib-model/SKILL.md`)의 10단계(단계 0~9) 절차를 그대로 따르며, 이 문서의 순서를 갱신한다.

## 7. 문서 갱신 규칙

모델을 포팅하면:
1. 이 표에서 해당 행을 "구현 완료" 표(§3)로 옮긴다.
2. `docs/guides/anomaly-models.md`를 `/add-anomalib-model` 절차에 따라 갱신한다(§7 문서 갱신 기준 참조).
3. `docs/dev/v0.2/reports/UPSTREAM-INVENTORY.md`에 파일 인벤토리를 추가한다.
