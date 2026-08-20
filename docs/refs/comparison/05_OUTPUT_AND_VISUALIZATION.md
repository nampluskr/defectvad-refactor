# `defectvad`, `roi-corner-detection-ver3`, `cv_boilerplate`의 output과 visualization 비교

> **저장소 참조 안내** — 본문의 `<repo>@<commit>:<path>#<symbol>` 표기는 아래 깃허브 저장소를 가리킨다.
>
> | 표기 | 깃허브 경로 |
> |---|---|
> | `defectvad@14879ea2` | https://github.com/nampluskr/defectvad |
> | `cv_boilerplate@65d5412b` | https://github.com/nampluskr/cv_boilerplate |
> | `roi-corner-detection-ver3@8ae989a8` | https://github.com/nampluskr/roi-corner-detection-ver3 |
>
> 특정 파일은 `https://github.com/nampluskr/<repo>/blob/<commit>/<path>` 로 조회한다.


## 1. 목적

이 문서는 model raw output이 metric, threshold, 저장 artifact와 사용자 시각화로 이어지는 경로를 비교한다. Anomaly detection에서 image-level score와 pixel-level anomaly map은 같은 tensor가 아니며, score 산출·map 정렬·threshold 결정·metric protocol을 숨기지 않는 것이 reference 재현의 전제다.

데이터와 mask의 공간 contract는 [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md), raw output을 task output으로 정규화하는 adapter 경계는 [03_MODEL_AND_ADAPTER.md](03_MODEL_AND_ADAPTER.md), 실행 시점 및 checkpoint selection은 [04_EXECUTION_LIFECYCLE.md](04_EXECUTION_LIFECYCLE.md)에서 다룬다.

## 2. 분석 기준

분석 기준일은 2026-08-20이다. 정적 코드와 config만 확인했으며, metric 수치, threshold 또는 visualization image의 정확성을 실행으로 검증하지 않았다.

| 저장소 | Revision | output 비교에서의 역할 |
|---|---|---|
| `defectvad` | `14879ea2a8970cee25438500e5abfeeb4be8e358` | anomaly-specific score/map evaluator와 visualizer의 레거시 근거 |
| `roi-corner-detection-ver3` | `8ae989a88996441e44fb2d5296a6419a8f661220` | final prediction CSV와 scalar metrics 분리의 사용자 운용 근거 |
| `cv_boilerplate` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` | adapter-mediated metric, serializable prediction 및 visualization hook의 현재 근거 |

연결 요구사항은 `FR-014`, `FR-016`, `FR-018`~`FR-020`, `FR-022`, `FR-024`, `FR-025`, `NFR-001`, `NFR-002`, `NFR-008`, `NFR-011`, `CON-006`, `CON-009`, `CON-010`이며, 관련 완료 기준은 `AC-005`, `AC-006`, `AC-012`, `AC-013`, `AC-015`다.

## 3. 한눈에 보는 결론

```text
defectvad
model {pred_score, anomaly_map}
  -> Evaluator image/pixel metric + F1 threshold
  -> Visualizer overlay / saved image

roi-corner-detection-ver3
raw output -> postprocessor -> final corners
  -> Evaluator metrics.json
  -> Predictor predictions.csv

cv_boilerplate
adapter eval output -> task metric accumulator -> metrics artifact
adapter predict output -> serializable sample records
  -> optional adapter save_predictions / visualize
```

| 질문 | `defectvad` | `roi-corner-detection-ver3` | `cv_boilerplate` |
|---|---|---|---|
| canonical model output | `pred_score`, `anomaly_map` dict key | normalized 4-corner NumPy array | adapter-defined `outputs` / serializable sample records |
| metric 소유 | anomaly `Evaluator` | evaluator와 wrapper metric | adapter `reset/update/compute_metrics` |
| threshold | evaluator의 F1 solve, predict의 optional calibration | corner success/metric policy | adapter/model-specific final metric or artifact |
| scalar artifact | run script logging/result | `metrics.json`, `history.json` | metrics writer 및 command run output |
| prediction artifact | prediction dict와 image files | `predictions.csv` | adapter `save_predictions()` optional hook |
| visualization | anomaly map, threshold를 아는 `Visualizer` | 이 문서의 직접 비교 대상 아님 | adapter `visualize()` optional hook |

`권고` 목표 anomaly contract는 raw upstream tensor 자체가 아니라 명시적 semantic record여야 한다. 최소 record에는 sample identity, image-level score, optional pixel-level map reference 또는 저장 위치, spatial shape/transform provenance, threshold identity와 prediction status를 포함한다. metric summary는 protocol, split, threshold mode, checkpoint, config 및 model/source identity와 분리되지 않게 저장한다.

## 4. `defectvad`: evaluator와 visualizer가 anomaly output key를 직접 해석한다

`확인된 사실` `Evaluator.evaluate_image_level()`은 batch의 `label`과 model output의 `pred_score`를 flatten해 binary AUROC와 average precision을 update한다. 모든 score/label을 모은 뒤 F1 및 F1 threshold를 계산해 `auroc`, `aupr`, `f1`, `th`를 반환한다.

`확인된 사실` `Evaluator.evaluate_pixel_level()`은 batch `mask`와 model output의 `anomaly_map`을 flatten해 같은 방식으로 pixel-level metric과 threshold를 계산한다.

근거: `defectvad@14879ea2:src/defectvad/common/evaluator.py#Evaluator`

```python
scores = preds["pred_score"].flatten()
anomaly_map = preds["anomaly_map"]
labels = batch["label"]
masks = batch["mask"]
```

`확인된 사실` predict script는 test dataloader 결과를 `BaseModel.predict()`로 누적한 뒤 `Visualizer(preds)`를 만든다. 선택적으로 train loader를 이용해 image/pixel threshold를 calibration하고 visualizer에 설정한다. CLI에는 anomaly/normal 저장, image/pixel level, 최대 sample 수 option이 있다.

근거:

- `defectvad@14879ea2:experiments/predict.py#predict`
- `defectvad@14879ea2:src/defectvad/common/base_model.py#BaseModel.predict_dataloader`
- `defectvad@14879ea2:src/defectvad/common/visualizer.py#Visualizer`

`해석` 레거시는 image score와 pixel map을 구별하는 출발점으로 유용하다. 그러나 evaluator/visualizer가 key 이름, mask, label과 prediction dict 전체를 직접 해석하므로 model별 score reduction, map resize/smoothing, threshold source 및 artifact provenance가 common anomaly class에 결합된다. 또한 test evaluation의 F1 threshold와 predict-time train calibration은 서로 다른 threshold protocol일 수 있으므로 결과에 threshold origin을 명시하지 않으면 비교가 모호하다.

`이전 판정` image/pixel metric을 구분하고 정상 sample에도 zero mask를 전달하는 의미는 `조정`하여 유지한다. `pred_score`/`anomaly_map` 문자열 key를 일반 evaluator와 visualizer가 직접 읽는 구조는 `대체`한다. 저장된 기존 visualization 양식은 reference diagnosis가 필요할 때만 adapter-specific renderer에서 `재사용` 여부를 판단한다.

## 5. `roi-corner-detection-ver3`: final prediction과 scalar 결과를 분리한다

`확인된 사실` wrapper `predict_step()`은 postprocessor를 거친 final corner array를 반환한다. `Evaluator`는 이 prediction과 target으로 metric instance를 update하고 `metrics.json`에 scalar result를 저장한다. 기본 evaluator metric에는 polygon IoU, mean/max corner distance, PCK, success rate가 포함된다.

근거:

- `roi-corner-detection-ver3@8ae989a8:src/models/base/wrapper.py#BaseWrapper.predict_step`
- `roi-corner-detection-ver3@8ae989a8:src/core/evaluator.py#build_default_metrics`
- `roi-corner-detection-ver3@8ae989a8:src/core/evaluator.py#Evaluator.save`

`확인된 사실` `Predictor`는 각 sample에 index, prediction 유효성, failure reason, optional target과 4개 corner 좌표를 기록해 `predictions.csv`로 저장한다. NaN/inf prediction은 `invalid_prediction`으로 표시한다.

근거: `roi-corner-detection-ver3@8ae989a8:src/core/predictor.py#Predictor`

`해석` 이 프로젝트의 참고 가치는 final output을 sample-level machine-readable artifact로 저장하고 scalar evaluation과 분리한다는 점이다. corner geometry, success 판정 및 CSV 열은 anomaly의 canonical output이 아니다. 다만 anomaly workflow도 sample identity와 invalid/omitted status를 유지해야 오류를 성공 또는 정상 prediction으로 오인하지 않는다.

`사용자 작업 지점` 새 ROI postprocessor는 `src/models/<model>/postprocessor.py`, 결과 열/저장은 `src/core/predictor.py`, metric set은 `src/core/evaluator.py`에서 찾아 변경한다. 목표 anomaly workflow는 비슷하게 model package/adapter, metric registry 및 output writer의 책임을 나누되, 사용자가 resolved config와 artifact location을 한 실행에서 찾을 수 있게 해야 한다.

## 6. `cv_boilerplate`: semantic output은 adapter를 통해서만 전달한다

`확인된 사실` `TaskAdapter`는 eval output의 metric update/compute/reset, sample별 serializable prediction, `save_predictions`와 `visualize` hook을 정의한다. base class의 저장/시각화 hook은 no-op이므로 task가 필요할 때만 구현한다.

근거: `cv_boilerplate@65d5412b:src/core/adapter.py#TaskAdapter`

`확인된 사실` `Trainer.evaluate()`는 adapter eval output을 받아 metric을 갱신한 후 `compute_metrics()` result를 반환한다. `Trainer.predict()`는 각 batch의 adapter prediction list를 하나의 list로 모은다. CLI evaluate는 config가 요청한 경우 batch별 `adapter.predict_step()`과 `adapter.visualize()`를 호출한다.

근거:

- `cv_boilerplate@65d5412b:src/core/engine.py#Trainer.evaluate`
- `cv_boilerplate@65d5412b:src/core/engine.py#Trainer.predict`
- `cv_boilerplate@65d5412b:src/cli/commands.py#evaluate`
- `cv_boilerplate@65d5412b:src/cli/commands.py#predict`

`확인된 사실` classification 및 segmentation adapter는 각각 task-specific visualize module을 import해 grid를 저장한다. segmentation adapter는 JSON-serializable predict return과 visualization에 필요한 tensor cache를 분리한다.

근거:

- `cv_boilerplate@65d5412b:src/tasks/classification/adapter.py#ClassificationAdapter.visualize`
- `cv_boilerplate@65d5412b:src/tasks/segmentation/adapter.py#SegmentationAdapter.visualize`

`해석` adapter boundary는 metric/renderer가 model raw output을 직접 가정하지 않도록 만든다. 하지만 base `predict_step`의 serializable-list 요구만으로 large anomaly map을 모두 JSON에 넣는 것은 적절하지 않다. map tensor는 per-sample file 또는 visualization artifact로 저장하고 prediction summary에는 위치, shape, dtype, normalization 및 map 생성 protocol을 남기는 쪽이 더 안전하다.

## 7. 목표 anomaly output contract

`권고` 아래는 구현 확정 전 문서용 contract 초안이다. field 이름과 file format은 P1/P2 source audit 뒤 확정하며, 이 표 자체를 현 구현 사실로 해석하지 않는다.

| 산출물 | 필수 내용 | 책임 | 금지 또는 주의 |
|---|---|---|---|
| sample prediction record | sample ID/path, image score, status, checkpoint/config identity | anomaly adapter | map tensor를 무조건 JSON inline으로 저장하지 않음 |
| pixel-map artifact | map path, shape, dtype, resize/normalization 및 source output identity | adapter output writer | image와 mask 좌표계 불일치 숨김 금지 |
| image metric summary | split, AUROC/AP/F1 등 승인 metric, threshold mode/value | metric adapter | test-derived selection 값을 train/valid selection으로 사용 금지 |
| pixel metric summary | mask contract, spatial alignment, metric, threshold mode/value | metric adapter | missing mask를 정상 mask로 silent fallback 금지 |
| visualization | input, map/overlay, threshold/decision, ground truth availability | adapter renderer | metric용 map과 다른 undocumented map 사용 금지 |
| provenance manifest | source/license, local asset, resolved config, checkpoint, protocol, exit status | command/benchmark layer | artifact만 남기고 실행 조건을 잃지 않음 |

```text
raw model output
  -> adapter semantic normalization
  -> metric accumulator (approved split/protocol)
  -> sample record + optional map artifact
  -> visualization renderer
  -> provenance-linked result summary
```

`권고` threshold는 다음 중 하나를 결과에 명시한다: fixed reference value, train-only calibration, valid-selected calibration, threshold 미사용 ranking metric. Test label로 threshold를 찾는 exploratory report와 acceptance report는 동일 artifact 이름으로 혼합하지 않는다. Pixel map이 resize, blur, Gaussian smoothing 또는 reduction을 거쳤다면 모든 parameter를 resolved config와 result manifest에 기록한다.

## 8. 미결정 사항과 검증

- 각 승인 model의 `pred_score` 정의와 `anomaly_map` 정의, score-map 일관성 및 batch shape는 pinned anomalib source에서 확인해야 한다.
- image/pixel metric set, threshold selection data, tolerance와 repeat policy는 `P0-T02`, `P0-T05` 및 SPEC §11의 protocol 승인 없이는 확정할 수 없다.
- map artifact의 file format, compression, retention 수와 visualization layout은 local storage budget 및 사용자 review workflow에 맞춰 결정해야 한다.
- reference comparison은 scalar metric만이 아니라 sample score, map, threshold 및 visualization input을 승인된 tolerance로 대조해야 한다.

이 검증이 끝나기 전에는 toy task adapter의 output 형태나 ROI corner CSV를 anomaly production contract로 승격하지 않는다.

작성일: 2026-08-20  
상태: 세 저장소 정적 비교 초안
