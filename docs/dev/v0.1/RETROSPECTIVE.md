# RETROSPECTIVE — v0.1 구조 회고 및 v0.2 진입 판단

상위 문서: [PLAN.md](PLAN.md) §4 · [SPEC.md](SPEC.md) §7 · 근거: [AC-verification.md](AC-verification.md)

P6-T03 산출물이다. v0.1에서 확립된 추상화가 v0.2(데이터셋 확장)를 **재설계 없이** 수용하는지 검토하고, SPEC §7의 미결정 항목을 해소하거나 이월한다. PLAN §2의 기준에 따라 "확장 과정에서 공통 구조를 다시 설계해야 한다면 v0.1의 추상화가 부족했다는 신호"로 본다.

## 1. 판정

**v0.2 진입 가능하다.** 조건부다 — 아래 §5의 이월 항목 두 건을 v0.2 범위에 포함하되, 둘 다 데이터셋 확장을 막지 않으므로 진입의 선행 조건은 아니다.

| 축 | 판정 | 근거 |
|---|---|---|
| 데이터셋 추상화 | 수용 가능 | §3.1 |
| 모델·adapter 경계 | 수용 가능 | §3.2 |
| 공통 engine 오염 | 없음 | AC-010 |
| 자산 경로 이식성 | 수용 가능 | §3.3 |
| 성능 판정 체계 | 미완 1건 (AC-006) | §5.1 |

## 2. v0.1이 실제로 검증한 것

의도한 두 축이 모두 검증되었다.

- **lifecycle 폭** — STFPM(단순 gradient training)과 EfficientAD(auxiliary 스트림 + teacher 통계 + 매 epoch 분위수 보정)를 **같은 `core/engine.py`, 같은 `Trainer.fit` 시그니처**로 구동했다. EfficientAD 전용 개념은 `core/`에 한 줄도 들어가지 않았다(AC-007·AC-010).
- **upstream 불가침** — 9개 파일에서 import 외 차이 0라인을 유지한 채 두 모델을 통합했다(AC-001). 성능 문제(P3·P5) 조사도 전부 adapter·config 쪽에서 해결했다.

구조적으로 유효했던 결정 네 가지를 기록한다. v0.3 이후에도 유지한다.

1. **wrapper가 아니라 adapter가 모델 차이를 흡수한다**(SPEC §4.1). `MODELS`에 등록되는 것은 팩토리 함수이고 반환물은 upstream `nn.Module` 원본이다. 모델 자리에 프로젝트 코드가 한 겹도 끼지 않는다.
2. **`InferenceBatch` 변환을 `postprocess.py#to_output_dict` 한 곳에 둔 것**(SPEC §4.2). 반환 타입·shape로만 판단하므로 모델이 늘어도 adapter마다 반복되지 않는다.
3. **auxiliary 스트림을 adapter가 자체 조달한 것**(SPEC §4.5). engine에 세 번째 loader를 추가했다면 나머지 4개 task가 쓰지 않는 인자를 떠안았다.
4. **`used_placeholder_keys()`로 참조되지 않는 path key를 검증에서 제외한 것**(SPEC §6.3). 백본이 없는 `custom_*` 모델이 `backbone_root` 부재로 막히지 않는다.

## 3. v0.2 수용 가능성 검토

### 3.1 데이터셋 확장 (BTAD, VisA)

v0.2가 요구하는 변경 범위를 실제 코드로 확인했다.

| 요소 | v0.2에서 필요한 일 | 재설계 필요 여부 |
|---|---|---|
| `DATASETS` registry | `@DATASETS.register("btad_anomaly")` 등 추가 | 없음 |
| 데이터셋 계약 | train `(image, {})`, valid/test `(image, {"label", "mask"})` — MVTec 고유 개념이 계약에 없다 | 없음 |
| split | `configs/splits/*.json` + `scripts/generate_*_splits.py` 추가. `src/data/split.py`는 id 문자열만 다루고 디렉터리 구조를 모른다 | 없음 |
| transform | `anomaly_default`가 image_size와 `normalize` 플래그만 받는다 | 없음 |
| metric | `image_auroc`/`pixel_auroc`가 label·mask 텐서만 본다 | 없음 |
| adapter | 변경 없음 | 없음 |
| `upstream/` | 변경 없음 | 없음 |
| config | `data.name`·`data.root`·`data.split.path`·`data.params.category` 교체 | 없음 |

MVTec 결합이 남아 있는 지점은 `MVTecAnomaly` 클래스 **내부**뿐이다(`train/good`, `test/<defect_type>`, `ground_truth/<...>_mask.png` 경로 규칙과 `train_good` id 접두사). 이는 클래스 안에 갇혀 있고 계약 밖으로 새지 않으므로, 새 데이터셋은 같은 계약을 만족하는 별도 클래스를 추가하면 된다. **재설계 신호 없음.**

다만 v0.2에서 실제로 부딪힐 지점 두 가지를 미리 적는다. 둘 다 새 데이터셋 클래스 안에서 흡수 가능하다.

- **VisA의 split 정의 방식이 다르다.** VisA는 CSV로 split을 배포한다. 이 프로젝트의 계약은 "id 리스트 JSON"이므로, 변환은 `scripts/generate_visa_splits.py`가 담당한다. `src/data/split.py`를 고칠 이유는 없다.
- **BTAD의 mask 파일 확장자·명명이 다르다.** 경로 규칙은 데이터셋 클래스의 `_mask_path`에 해당하는 사적 메서드가 흡수한다.

### 3.2 모델·adapter 경계

v0.2는 모델을 늘리지 않으므로 직접적인 부담이 없다. v0.3을 대비해 확인한 결과, 현재 hook 집합(`on_fit_start` / `on_epoch_start` / `on_validation_start` / `on_epoch_end` / `on_fit_end` / `extra_final_metrics`)은 EfficientAD가 요구한 모든 시점을 덮었다. gradient training이 없는 모델(patchcore, padim 등)은 "fit 후 bank 구축"을 `on_fit_end`로, "epoch 0회 학습"을 `train.epochs`로 표현할 수 있을 것으로 보이나, **v0.3의 실제 시험대이며 지금 선제적으로 추상화하지 않는다**(두 모델 이상에서 확인되기 전 새 추상화 금지).

### 3.3 자산 경로 이식성

`${paths.*}` placeholder와 4단계 해석 우선순위는 데이터셋이 늘어도 그대로 동작한다. v0.2에서 할 일은 `configs/assets.yaml`에 BTAD·VisA 항목을 추가하는 것뿐이다.

## 4. SPEC §7 미결정 항목 처리

| 항목 | 처리 |
|---|---|
| 기존 hook만으로 EfficientAD lifecycle이 충분한지 | **해소.** 충분하지 않았고, P5에서 task-agnostic hook `on_validation_start`를 `core/adapter.py`·`core/engine.py`에 추가해 해결했다. 기본 no-op이므로 나머지 4개 task와 STFPM은 영향 없다(AC-010) |
| auxiliary transform이 `transform.py`에 별도로 필요한지 | **해소.** 별도 transform은 불필요했다. auxiliary 전용 파이프라인(Resize×2 → RandomGrayscale → CenterCrop → ToTensor)은 anomalib과 동일하게 `EfficientAdAdapter._build_auxiliary_loader` 안에 두었고, 공통 `anomaly_default`에는 `normalize` 플래그만 추가했다 |
| MVTec 대표 3개 카테고리 선정 | **해소.** bottle, carpet, capsule (P3-T01) |
| 복수 parameter group을 요구하는 모델의 optimizer 표현 | **v0.3으로 이월.** v0.1·v0.2 두 모델은 freeze 결과가 곧 optimizer 대상 집합이라 `build_optimizer`의 단일 group으로 충분하다. 해당 모델을 만나는 시점에 `core/builders.py`에 group 지정 방식을 추가한다(공통 코드 변경 등급 C) |

## 5. v0.2로 이월하는 항목

### 5.1 AC-006 — EfficientAD reference 재현 미충족

capsule image AUROC 0.683(reference 0.982)으로 미충족이다. 구현 결함은 P5에서 배제되었고, 남은 원인은 학습 예산(reference의 6~8% 스텝)과 capsule valid 정상 이미지 9장에서 오는 추정 분산이다.

- 조치: `train.epochs`를 reference에 근접한 수준으로 올리고(`optim.scheduler.params.step_size = int(0.95 * epochs)`를 함께 변경) capsule을 재측정한다.
- **이 항목은 v0.2 진입을 막지 않는다.** 데이터셋 추상화 검증과 독립이며, 원인이 구조가 아니라 실행 예산이기 때문이다. 다만 v0.2에서 데이터셋이 늘면 예산 문제가 카테고리 수만큼 커지므로, v0.2 착수 시 먼저 처리하는 것을 권한다.

### 5.2 threshold 영속화

`AnomalyAdapter.on_fit_end`가 계산한 image/pixel threshold가 adapter 인스턴스와 `metrics_final.json`에만 남고 checkpoint에 저장되지 않는다. 그 결과 standalone predict의 `is_anomalous`·`image_threshold`가 `null`이다(AC-005 단서).

- 두 모델 공통이며 P2 시점부터 존재한 기존 한계다.
- 조치 방향: `on_fit_end`가 threshold를 모델 buffer 또는 checkpoint 메타로 남기고 predict 경로가 읽는다. 공통 `AnomalyAdapter` 변경이므로 모델명 분기 없이 구현 가능하다.
- v0.2에서 데이터셋이 늘면 threshold를 데이터셋·카테고리별로 관리해야 하므로, 이 시점에 해결하는 편이 비용이 낮다.

### 5.3 monitor metric 포화

`core/engine.py#Trainer.fit`의 갱신 조건이 엄격한 부등호(`current > best_metric`)이므로, `image_auroc`가 1.0에 도달하는 카테고리(bottle, carpet)에서는 이후 갱신이 일어나지 않고 `best.pth`가 그 시점에 고정된다. `PRD.md` §4에 이월 항목으로 기록되어 있다. v0.2에서 데이터셋이 늘어도 같은 현상이 재현되므로 함께 검토한다.

## 6. v0.2 착수 시 첫 작업 제안

순서만 제안하며 범위는 v0.2 착수 시 확정한다.

1. §5.1 — EfficientAD 학습 예산을 올려 capsule 재측정, AC-006 해소.
2. §5.2 — threshold 영속화 (데이터셋이 늘기 전).
3. BTAD 데이터셋 클래스 + split 스크립트 + config → 두 모델 스모크.
4. VisA 동일.
5. 데이터셋 추가 절차를 [MODEL-ADD.md](MODEL-ADD.md) §4에 실제 경험으로 보정.

---

작성일: 2026-08-21
문서 상태: P6-T03 산출물 — v0.2 진입 가능 판정
