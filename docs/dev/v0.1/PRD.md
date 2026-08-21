# PRD — Anomaly Detection Integration on `cv_boilerplate`

상위 문서: [BRIEF.md](BRIEF.md) · 하위 문서: [PLAN.md](PLAN.md)

이 문서는 BRIEF의 의도와 세 원칙을 **검증 가능한 요구사항**으로 변환한다. 범위는 v0.1(STFPM, EfficientAD, MVTec AD)이다.

## 1. 기능 요구사항 (FR)

| ID | 요구사항 | 근거 |
|---|---|---|
| FR-001 | anomalib `models/image/<model>/`의 순수 PyTorch 파일을 수정 없이 복사해 모델을 구성한다 | 원칙 1 |
| FR-002 | `lightning_model.py`의 optimizer·scheduler·post-processing·hooks를 wrapper로 재구성한다 | 원칙 2 |
| FR-003 | 공통 인터페이스로 `train`을 실행한다 | G-01, G-04 |
| FR-004 | 공통 인터페이스로 `evaluate`를 실행하고 image/pixel AUROC를 산출한다 | G-01, G-04 |
| FR-005 | 공통 인터페이스로 `predict`를 실행하고 anomaly score와 map을 산출한다 | G-01, G-04 |
| FR-006 | MVTec AD 데이터셋을 카테고리 단위로 로딩한다 | v0.1 범위 |
| FR-007 | 학습된 모델과 lifecycle state를 checkpoint로 저장·복원한다 | G-09 |
| FR-008 | 데이터셋·가중치 경로를 config로 지정한다 | 원칙 3 |
| FR-009 | STFPM을 통합한다 (SGD lr 0.4, momentum 0.9, weight_decay 0.001, 스케줄러 없음) | v0.1 범위 |
| FR-010 | EfficientAD를 통합한다 (Adam lr 1e-4, weight_decay 1e-5, StepLR 95% 시점 0.1배) | v0.1 범위 |
| FR-011 | EfficientAD의 auxiliary 데이터(ImageNette), teacher 채널 통계, 검증 전 분위수 계산을 wrapper에서 수행한다 | FR-010 |
| FR-012 | 실행 결과와 재현 정보(config, seed, 경로)를 기록한다 | G-09 |

## 2. 비기능 요구사항 (NFR)

| ID | 요구사항 |
|---|---|
| NFR-001 | anomalib reference와 비교 가능한 성능을 재현한다 (판정은 §5) |
| NFR-002 | 동일 config·seed로 결과를 재현할 수 있다 |
| NFR-003 | 새 모델 추가 시 공통 engine을 복제·재작성하지 않는다 |
| NFR-004 | 데이터셋 인터페이스가 MVTec 디렉터리 구조에 결합되지 않는다 |
| NFR-005 | 공통 engine에 모델명·task명 기반 분기를 두지 않는다 |
| NFR-006 | 인터넷 연결 없이 전체 lifecycle을 실행한다 |
| NFR-007 | 압축 배포된 저장소를 다른 로컬 환경에서 저장소 파일 수정 없이 실행한다. 머신별 자산 경로는 `configs/local.yaml` 또는 환경변수로 지정하며, 미지정·부재 시 즉시 실패한다 (SPEC §6) |

## 3. 제약 (CON)

| ID | 제약 | 위반 시 |
|---|---|---|
| CON-001 | `torch_model.py` 등 복사한 모델 코드는 import 경로 외 수정 금지 | 즉시 되돌린다 |
| CON-002 | Lightning을 의존성 추가·import·호출하지 않는다 | 즉시 제거한다 |
| CON-003 | config가 가리키는 로컬 자산만 사용한다. 경로는 머신별로 지정 가능하며(SPEC §6) 개발 머신 기본값은 `/mnt/d/datasets`·`/mnt/d/backbones`다 | `configs/local.yaml` 또는 config를 수정한다 |
| CON-004 | AI 에이전트는 데이터셋·모델·라이브러리를 자동 다운로드·설치하지 않는다 | 사용자에게 CLI 설치를 요청하고 대기한다 |
| CON-005 | 대상은 image 모델로 한정한다 (video 제외) | 범위에서 제외한다 |

## 4. 범위 외 (v0.1)

- MVTec 외 데이터셋 (BTAD, VisA) → v0.2
- STFPM·EfficientAD 외 모델 → v0.3, v0.4
- 15개 카테고리 전체 벤치마크 → v0.2 이후
- 분산 학습, MLOps, 배포 기능
- **모델 선택 정책의 monitor metric 포화 대응 → v0.2** (2026-08-21 P5에서 발견, 사용자 판단으로 이월)

### monitor metric 포화 (v0.2 이월)

`core/engine.py#Trainer.fit`의 갱신 조건이 엄격한 부등호(`current > best_metric`)이므로, monitor metric이 상한에 도달하면 이후 갱신이 일어나지 않고 `best.pth`가 그 시점에 고정된다. AUROC·accuracy처럼 최댓값이 1.0인 지표에서 발생한다.

P5 재측정에서 실제로 관측되었다. bottle은 `image_auroc`가 epoch 3에 1.0에 도달해 `best.pth`가 고정되는 동안, 감시 대상이 아닌 `pixel_auroc`는 epoch 20까지 0.922 → 0.985로 계속 개선되었다. 보고된 bottle `pixel_auroc` 0.922는 모델 성능이 아니라 선택 정책의 산물이며 실제 도달값보다 0.063 낮다.

v0.1에서 다루지 않는 이유는 `core/`의 공통 선택 정책이라 anomaly 외 4개 task(classification, detection, segmentation, toy)에 모두 영향을 주는 등급 C 변경이고, 동점 처리 정책을 어떻게 정할지가 v0.1 범위를 넘는 설계 판단이기 때문이다.

v0.2에서 검토할 방향은 다음과 같다. 어느 쪽이든 공통 루프에 모델명·task명 분기를 두지 않아야 한다(NFR-005).

- 동점 시 후순위 epoch 채택(`>=`) — 단순하나 과적합이 진행된 후반 epoch을 고르는 위험이 있다
- 동점 시 보조 지표로 판정 — 타당하나 "복수 지표 기반 선택"이라는 새 개념을 공통 루프에 도입해야 한다
- 포화하지 않는 monitor metric 사용(average precision 등) — reference와의 비교 기준이 달라진다

v0.1 결과 해석 시에는 bottle `pixel_auroc`가 과소 보고된 값이라는 단서를 함께 읽는다.

## 5. 성능 판정 절차

성능 tolerance를 문서에서 미리 고정하지 않는다. **사용자가 터미널에서 직접 실행하고 그 결과를 피드백**하여 판정한다.

### 5.1 절차

1. 에이전트가 코드 구현을 완료한다.
2. 에이전트는 **사용자가 터미널에서 실행할 명령어를 제시**한다.
3. 사용자가 실행하고 결과(AUROC 등)를 피드백한다.
4. 에이전트는 결과를 anomalib reference와 비교하고, 차이가 있으면 원인(§5.4)을 검토해 수정한다.
5. 사용자가 재현 성공으로 판단하면 해당 Phase를 완료한다.

에이전트는 학습·평가를 임의로 장시간 실행하지 않는다. 실행 주체는 사용자다.

### 5.2 anomalib reference 값 (MVTec 15개 카테고리 평균)

| 모델 | 조건 | image AUROC | pixel AUROC |
|---|---|---|---|
| STFPM | ResNet-18 | 0.893 | 0.951 |
| STFPM | Wide ResNet-50 | 0.876 | 0.903 |
| EfficientAD-S | batch size 1 | 0.982 | — |
| EfficientAD-M | batch size 1 | 0.975 | — |

출처: anomalib 각 모델 README. 카테고리별 값은 원문을 참조한다.

### 5.3 검증 범위

| 단계 | 범위 | 목적 |
|---|---|---|
| 개발 중 | 대표 1개 카테고리 (예: bottle) | 빠른 반복, 동작 확인 |
| Phase 완료 | 대표 3개 카테고리 | reference 대비 성능 판정 |

전체 15개 카테고리 평균 비교는 v0.2 이후로 미룬다.

### 5.4 차이 발생 시 검토 항목

성능 차이를 모델 구현 차이로 단정하지 않는다. 다음을 먼저 확인한다.

preprocessing · augmentation · optimizer/scheduler 설정 · epoch/batch 조건 · pretrained weight · score normalization · post-processing · threshold 계산 · metric 구현 · evaluation protocol

pretrained weight는 특히 STFPM에서 우선 검토 대상이다. anomalib은 timm `resnet18`의 기본 pretrained를, 이 프로젝트는 `/mnt/d/backbones`의 torchvision 가중치를 쓰므로 학습 레시피가 다르다([SPEC.md](SPEC.md) §4.6).

원인이 무엇이든 **모델 코드는 수정하지 않는다**(CON-001). wrapper 또는 boilerplate에서 해결한다.

## 6. 완료 조건 (AC)

| ID | 조건 | 검증 |
|---|---|---|
| AC-001 | 복사한 모델 코드가 import 경로 외에 원본과 동일하다 | upstream과 diff |
| AC-002 | 코드베이스에 Lightning 의존이 없다 | 의존성·import 검색 |
| AC-003 | STFPM의 train/evaluate/predict가 동작한다 | 사용자 실행 |
| AC-004 | STFPM이 reference 성능을 재현한다 | §5 절차, 대표 3개 카테고리 |
| AC-005 | EfficientAD의 train/evaluate/predict가 동작한다 | 사용자 실행 |
| AC-006 | EfficientAD가 reference 성능을 재현한다 | §5 절차, 대표 3개 카테고리 |
| AC-007 | EfficientAD의 auxiliary·통계 처리가 wrapper에 있다 | 코드 검토 |
| AC-008 | 인터넷 차단 상태에서 전체 lifecycle이 실행된다 | 오프라인 실행 |
| AC-009 | 동일 config·seed로 결과가 재현된다 | 반복 실행 비교 |
| AC-010 | 두 모델이 공통 engine을 공유하며 모델명 분기가 없다 | 코드 검토 |

---

작성일: 2026-08-20
문서 상태: Initial PRD
