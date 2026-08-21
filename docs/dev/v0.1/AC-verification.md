# v0.1 완료 조건(AC) 검증표

`PRD.md` §6의 AC-001~AC-010을 전수 검증한 기록이다(P5-T03).

- 검증 일시: 2026-08-21
- 대상 코드: `bdb131b` 이후 작업 트리 (P5 calibration 수정 반영)
- 검증 주체: 메인 세션. 성능 판정(AC-004, AC-006)은 사용자 실행 결과와 사용자 판단에 따른다(PRD §5.1).

## 결과 요약

| ID | 조건 | 판정 |
|---|---|---|
| AC-001 | 복사한 모델 코드가 import 경로 외에 원본과 동일하다 | 충족 |
| AC-002 | 코드베이스에 Lightning 의존이 없다 | 충족 |
| AC-003 | STFPM의 train/evaluate/predict가 동작한다 | 충족 |
| AC-004 | STFPM이 reference 성능을 재현한다 | 충족 |
| AC-005 | EfficientAD의 train/evaluate/predict가 동작한다 | 충족 (단서 있음) |
| AC-006 | EfficientAD가 reference 성능을 재현한다 | **미충족** |
| AC-007 | EfficientAD의 auxiliary·통계 처리가 wrapper에 있다 | 충족 |
| AC-008 | 인터넷 차단 상태에서 전체 lifecycle이 실행된다 | 충족 |
| AC-009 | 동일 config·seed로 결과가 재현된다 | 충족 |
| AC-010 | 두 모델이 공통 engine을 공유하며 모델명 분기가 없다 | 충족 |

10개 중 9개 충족, 1개(AC-006) 미충족이다.

## AC-001 — upstream 원본 동일성

`/mnt/d/projects/clones/anomalib`(commit `091ca6a`, v2.3.0)의 원본과 `src/tasks/anomaly/upstream/` 전 파일을 diff했다. import 문을 제외한 차이는 **모든 파일에서 0라인**이다.

| 파일 | import 외 차이 라인 수 |
|---|---|
| `efficient_ad/torch_model.py` | 0 |
| `stfpm/torch_model.py` | 0 |
| `stfpm/loss.py` | 0 |
| `stfpm/anomaly_map.py` | 0 |
| `components/tiler.py` | 0 |
| `components/feature_extractors/timm.py` | 0 |
| `components/feature_extractors/utils.py` | 0 |
| `components/data/generic.py` | 0 |
| `components/data/torch_base.py` | 0 |

변경된 import는 `anomalib.*` → 프로젝트 로컬 경로 치환뿐이다(`InferenceBatch`, `TimmFeatureExtractor`, `Tiler`).

## AC-002 — Lightning 의존 부재

- `requirements.txt`에 `lightning`/`pytorch_lightning` 없음.
- 실행 환경에 `lightning`, `pytorch_lightning`, `anomalib` 모두 미설치(`importlib.util.find_spec` 확인).
- 소스 전체 검색 결과 남은 4건은 모두 `upstream/`의 docstring 텍스트(`efficient_ad/torch_model.py:30-31`, `stfpm/loss.py:36,79`)로 실행 경로가 아니다. 이를 지우는 것은 CON-001 위반이므로 그대로 둔다.

## AC-003 — STFPM train/evaluate/predict 동작

`outputs/runs/anomaly/` 하위 산출물로 확인했다.

- train: `stfpm_bottle`, `stfpm_carpet`, `stfpm_capsule`
- evaluate(test split): `stfpm_*_eval__test` 3건
- predict: `p2_smoke_predict/predictions/predict.json`

## AC-004 — STFPM reference 재현

| 카테고리 | image AUROC | pixel AUROC |
|---|---|---|
| bottle | 1.0000 | 0.9598 |
| carpet | 1.0000 | 0.9931 |
| capsule | 0.8783 | 0.9645 |
| 평균 | 0.9594 | 0.9725 |
| reference (ResNet-18, 15개 평균) | 0.893 | 0.951 |

3개 카테고리 평균이 image·pixel 양쪽에서 reference를 상회한다. P3에서 사용자가 재현 성공으로 판단했다.

STFPM은 `on_validation_start` 기본 no-op을 상속하므로 P5의 calibration 수정에 영향받지 않는다. P3 측정값이 그대로 유효하다.

## AC-005 — EfficientAD train/evaluate/predict 동작

- train: `efficientad_{bottle,carpet,capsule}_fixed` 3건 정상 완주.
- evaluate(test split): `ac_ead_eval__test` — bottle test에서 image 0.9474 / pixel 0.9005. valid(1.0/0.9221)보다 낮은 것은 valid가 모델 선택·calibration·threshold에 쓰였기 때문으로 정상이다.
- predict: `ac_ead_predict__predict/predictions/predict.json` — 20건 생성, `anomaly_score` 정상 출력.

**단서**: predict 결과의 `is_anomalous`와 `image_threshold`가 `null`이다. threshold는 `AnomalyAdapter.on_fit_end`가 계산해 adapter 인스턴스에 두고 `metrics_final.json`에만 기록되며, checkpoint에는 저장되지 않는다(`best.pth` 키에 threshold 관련 항목 없음). 따라서 standalone predict는 이진 판정을 낼 수 없고 원시 점수만 제공한다.

이는 두 모델 공통이며 P2 시점부터 존재한 기존 한계다(`p2_smoke_predict`도 동일). AC-005의 조건은 "동작한다"이므로 충족으로 판정하되, threshold 영속화는 후속 과제로 남긴다.

## AC-006 — EfficientAD reference 재현 (미충족)

| 카테고리 | image AUROC | pixel AUROC | reference (image) |
|---|---|---|---|
| bottle | 1.0000 | 0.9221 | 0.982 |
| carpet | 0.9773 | 0.9574 | 0.982 |
| capsule | 0.6825 | 0.9444 | 0.982 |
| 평균 | 0.8866 | 0.9413 | 0.982 |

bottle·carpet은 reference 수준에 도달했으나 **capsule이 0.683으로 크게 미달**하여, 사용자 판단에 따라 미충족으로 기록한다.

### 원인 진단

구현 결함은 배제되었다. 저장된 `best.pth`로 평가만 수행해 image score 집계 방식 5종을 비교한 결과, 현재 방식(upstream `forward()`의 `amax_raw`)이 이미 최선이며 어떤 대안도 capsule을 구제하지 못했다(`A5.md` 참조). PRD §5.4의 post-processing·metric 항목은 원인에서 배제된다. P5에서 발견된 calibration 결함은 별도로 수정되었고 capsule은 그 수정으로 0.616 → 0.683으로 개선되었다.

남은 원인은 두 가지다.

1. **학습 예산**: 카테고리당 4,380~5,600 스텝으로 anomalib이 reference를 산출한 70,000 스텝의 약 6~8%다. 이 예산은 P4-T04에서 동작 확인용으로 정한 값이다.
2. **추정 분산**: capsule valid의 정상 이미지가 9장뿐이라 정상 1장의 순위 이동만으로 AUROC가 최대 `42/378 = 0.111` 움직인다. 관측된 epoch별 0.32~0.68 진동의 상당 부분이 모델 불안정이 아니라 추정 잡음이다.

### 재확인 방법

`train.epochs`를 reference에 근접한 수준으로 늘려(그에 맞춰 `optim.scheduler.params.step_size = int(0.95 * epochs)`도 함께 변경) capsule을 재실행한다. 이 검증은 v0.2로 이월한다.

## AC-007 — auxiliary·통계 처리의 위치

EfficientAD 전용 lifecycle이 모두 adapter에 있고 upstream과 공통 engine 어느 쪽에도 새어 나가지 않았다.

- adapter(`adapters/efficientad.py`): teacher 가중치 로드(`on_fit_start`), auxiliary loader 생성(`_build_auxiliary_loader`), teacher 채널 통계(`_teacher_channel_mean_std`), 분위수 calibration(`_map_norm_quantiles`, `on_validation_start`/`on_fit_end`).
- `upstream/efficient_ad/torch_model.py`: lifecycle hook·데이터 로딩 코드 없음(원본 그대로).
- `core/engine.py`: `auxiliary`/`imagenette` 등 관련 개념을 전혀 모름.

## AC-008 — 오프라인 실행

- `configs/anomaly/_base.yaml`의 `runtime.allow_network: false`.
- 오프라인 가드는 프로세스 시작 시 무조건 무장되고, config가 명시적으로 opt-in할 때만 해제된다(`cli/commands.py#apply_network_policy`). 가드는 `socket.connect`/`connect_ex`/`sendto`/`sendmsg`/`getaddrinfo`를 소켓 레벨에서 후킹해 loopback 외 접근 시 `OfflineViolationError`를 낸다.
- 가드가 무장된 상태에서 train 2회, evaluate 1회, predict 1회가 모두 완주했다. 네트워크 접근 시도가 있었다면 예외로 중단되었을 것이다.

## AC-009 — 동일 config·seed 재현성

동일 config·seed로 bottle 2 epoch 학습을 2회(`ac_repro_r1`, `ac_repro_r2`) 실행했다.

| 항목 | r1 | r2 |
|---|---|---|
| epoch 1 train loss | 10.69313328117845 | 10.69313328117845 |
| epoch 2 train loss | 7.889317980223296 | 7.889317980223296 |
| epoch 1 valid image/pixel | 0.9950000047683716 / 0.8150076270103455 | 동일 |
| epoch 2 valid image/pixel | 1.0 / 0.8335393667221069 | 동일 |
| image_threshold | 0.19761516153812408 | 0.19761516153812408 |
| pixel_threshold | 0.047978393733501434 | 0.047978393733501434 |

loss·metric·threshold가 비트 단위로 동일하다. 차이는 `elapsed_sec`뿐이며 이는 측정 시간이라 무관하다.

보조 증거로, P5 수정 1차·2차 시점의 3-epoch 스모크(`p5_fix_smoke_bottle`, `p5_fix_smoke_bottle2`) epoch별 metric도 완전히 일치했다. calibration을 `torch.random.fork_rng`로 감싼 것이 valid split 크기가 학습 RNG를 교란하지 않도록 보장한다.

## AC-010 — 공통 engine 공유 및 모델명 분기 부재

- `core/*.py`의 조건 분기(`if`/`elif`)에 모델명·task명이 쓰인 곳이 없다. 검색된 2건(`core/offline.py:112-113`)은 docstring 설명문이다.
- `StfpmAdapter`와 `EfficientAdAdapter`가 모두 `AnomalyAdapter`를 상속하고, `cli/commands.py`가 동일한 `Trainer`로 두 모델을 구동한다.
- P5에서 추가한 `on_validation_start`도 task-agnostic hook이며 기본 no-op이다. `StfpmAdapter`와 나머지 4개 task는 이를 그대로 상속한다.

## 후속 과제

- **AC-006 미충족 해소**: 학습 예산을 늘린 capsule 재측정 → v0.2.
- **threshold 영속화**: standalone predict가 이진 판정을 낼 수 있도록 threshold를 checkpoint에 저장 → v0.2 검토.
- **monitor metric 포화**: `PRD.md` §4에 v0.2 이월 항목으로 기록됨.
