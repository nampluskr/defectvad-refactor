# `defectvad`, `roi-corner-detection-ver3`, `cv_boilerplate`의 model과 adapter 비교

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

이 문서는 model을 만들고 학습 가능한 단계로 감싸는 경계, loss와 optimizer의 소유권, raw output을 task output으로 바꾸는 위치를 비교한다. 데이터의 sample 및 batch contract는 [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md), train/evaluate/predict 실행 순서는 [04_EXECUTION_LIFECYCLE.md](04_EXECUTION_LIFECYCLE.md), metric·threshold·시각화 output은 [05_OUTPUT_AND_VISUALIZATION.md](05_OUTPUT_AND_VISUALIZATION.md)에서 다룬다.

이 문서의 핵심 질문은 anomalib의 pure-PyTorch 알고리즘을 보존하면서도 `cv_boilerplate` 공통 engine에 anomaly task명 또는 특정 model명 분기를 넣지 않을 수 있는가이다.

## 2. 분석 기준

분석 기준일은 2026-08-20이다.

| 구분 | 저장소 | Revision | 역할 |
|---|---|---|---|
| 레거시 근거 | `defectvad` | `14879ea2a8970cee25438500e5abfeeb4be8e358` | anomaly model별 wrapper와 trainer의 과거 연결 방식 |
| 사용자 운용 근거 | `roi-corner-detection-ver3` | `8ae989a88996441e44fb2d5296a6419a8f661220` | 서로 다른 network/head raw output을 corner contract로 정규화하는 방식 |
| 현재 구조 | `cv_boilerplate` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` | task-agnostic adapter와 공통 engine의 현재 extension point |

정적 코드와 config만 확인했으며 학습, 평가 또는 upstream parity 실행은 하지 않았다. 주요 연결 요구사항은 `FR-001`, `FR-003`~`FR-005`, `FR-018`, `FR-024`, `FR-025`, `NFR-003`, `NFR-009`, `CON-002`~`CON-005`이며, 관련 설계는 SPEC §6, §7, §9, §17.1 및 계획 `P1-T02`~`P1-T05`, `P2-T01`~`P2-T04`다.

## 3. 한눈에 보는 결론

```text
defectvad
YAML module/class -> BaseModel + model별 BaseTrainer -> model dict output

roi-corner-detection-ver3
명시적 get_wrapper -> model + preprocessor + postprocessor + optimizer -> normalized corner

cv_boilerplate
registry model + TaskAdapter -> train/eval/predict hook -> serializable prediction
```

| 책임 | `defectvad` | `roi-corner-detection-ver3` | `cv_boilerplate` |
|---|---|---|---|
| 구현 선택 | config의 Python module/class | `get_wrapper()`의 명시적 model 조건 | registry logical name |
| model wrapper | `BaseModel`이 device, predict, save/load를 소유 | `BaseWrapper`가 model, pre/postprocess, optimizer와 metric을 소유 | `TaskAdapter`가 model 밖에서 batch 의미를 제공 |
| 학습 단계 | model별 trainer의 `training_step` | wrapper의 `train_step` | adapter의 `train_step`이 grad loss와 scalar detail 반환 |
| raw output 변환 | model과 evaluator가 `pred_score`, `anomaly_map`을 가정 | model별 postprocessor가 final corners로 변환 | adapter가 task별 output과 serializable prediction을 정의 |
| 공통화 위험 | anomaly dict와 output key가 base class 전반에 확산 | ROI corner contract가 모든 wrapper의 기준 | adapter contract를 지키면 engine은 task/model을 모름 |

`권고` anomaly model은 upstream algorithm, 학습에 필요한 buffer와 model-specific state를 model package에 보존한다. batch 해석, auxiliary 입력 조달, loss 조합, fit 전후 calibration, raw output의 score/map 정규화는 anomaly adapter에 둔다. optimizer를 epoch 중 교체해야 하는 model만 adapter hook을 통해 제한적으로 처리하며, engine 또는 CLI의 모델명 분기는 허용하지 않는다.

## 4. `defectvad`: anomaly wrapper와 trainer가 결합된 구조

`확인된 사실` `BaseModel`은 `torch.nn.Module`을 device로 옮기고 Tensor, batch dict, `DataLoader`를 입력 유형으로 구분해 prediction을 실행한다. dataloader prediction은 batch의 원래 key와 model prediction dict를 병합한 뒤 key별로 누적한다. weight file에는 model의 `state_dict`만 저장한다.

근거: `defectvad@14879ea2:src/defectvad/common/base_model.py#BaseModel`

```python
def predict_batch(self, batch):
    predictions = self.model(batch["image"].to(self.device))
    return {**batch, **predictions}
```

`확인된 사실` `BaseTrainer`는 optimizer, scheduler, device와 epoch state를 보유하고, subclass의 `training_step(batch)`을 호출한다. validation은 내장 `Evaluator.evaluate_image_level()`로 연결된다. STFPM 등의 model config는 model class와 trainer class의 Python 경로를 직접 지정한다.

근거:

- `defectvad@14879ea2:src/defectvad/common/base_trainer.py#BaseTrainer`
- `defectvad@14879ea2:configs/models/stfpm.yaml`

`해석` 레거시는 anomaly model 사이에서 반복되는 device 이동과 train loop를 줄였지만, base model/trainer/evaluator 모두가 anomaly batch dict와 `pred_score` 또는 `anomaly_map` key를 알고 있다. model의 algorithm 차이와 task semantics가 같은 계층에 있어 새로운 task 또는 model lifecycle을 추가할 때 base class를 바꿀 가능성이 커진다.

`이전 판정` `BaseModel`의 model-only save/load와 개별 model의 pure-PyTorch network/loss는 `조정` 대상이다. `BaseTrainer`의 공통 loop와 `Evaluator`에 고정된 output key 계약은 `대체` 대상이다. 원본 알고리즘의 forward, loss, buffer 및 update rule을 통째로 재설계하는 것은 `제외` 대상이다.

## 5. `roi-corner-detection-ver3`: wrapper로 raw output을 고정 contract에 맞춘다

`확인된 사실` `BaseWrapper`는 model, preprocessor, postprocessor, optimizer, scheduler, losses와 metrics를 소유한다. `train_step()`은 forward, loss 계산, backward와 optimizer step을 수행하고, `eval_step()`은 forward 및 metric update를 수행한다. `predict_step()`은 model raw output을 postprocessor에 전달해 CPU NumPy prediction을 반환한다.

근거: `roi-corner-detection-ver3@8ae989a8:src/models/base/wrapper.py#BaseWrapper`

```python
raw_output = self.model(images)
losses = self.compute_losses(raw_output, targets)
preds = self.postprocessor(raw_output)
```

`확인된 사실` concrete wrapper는 `DetWrapper`, `DetrWrapper`, `GCNWrapper`, `HybridWrapper`처럼 model-specific pre/postprocessor와 기본 metric을 선택한다. factory `get_wrapper()`는 사용자가 전달한 `model` option으로 wrapper를 명시적으로 고른다.

근거:

- `roi-corner-detection-ver3@8ae989a8:src/core/factory.py#get_wrapper`
- `roi-corner-detection-ver3@8ae989a8:src/models/det/wrapper.py#DetWrapper`
- `roi-corner-detection-ver3@8ae989a8:src/models/detr/wrapper.py#DetrWrapper`

`해석` 이 구조의 유용한 점은 서로 다른 raw output을 final corner라는 사용자-facing contract로 정규화한다는 점이다. 반면 optimizer step과 scheduler policy까지 wrapper가 실행하므로, 공통 engine이 AMP, clipping, checkpoint/resume 같은 실행 책임을 통일하기는 어렵다. ROI corner target과 postprocessor는 anomaly adapter의 공통 contract가 아니다.

`사용자 작업 지점` ROI에서 새 model을 추가할 때는 `src/models/<model>/`의 model/preprocessor/postprocessor/wrapper와 `src/core/factory.py#get_wrapper`를 함께 수정한다. 목표 architecture에서는 동일한 사용자 가시성을 registry registration, model package config 및 adapter parameter로 제공해야 하며, factory의 model-name 조건문을 공통 core에 복제하지 않는다.

## 6. `cv_boilerplate`: engine 밖의 `TaskAdapter`

`확인된 사실` `TaskAdapter`의 필수 contract는 `train_step`, `eval_step`, metric reset/update/compute, `predict_step`, `batch_size`다. `train_step`은 gradient가 연결된 `loss`와 기록용 `loss_dict`를 반환하며, `predict_step`은 sample별 serializable prediction list를 반환한다. `collate_fn`, fit/epoch hooks, prediction 저장 및 visualization hook은 선택 사항이다.

근거: `cv_boilerplate@65d5412b:src/core/adapter.py#TaskAdapter`

`확인된 사실` engine은 adapter를 통해서만 batch를 model에 전달한다. 따라서 engine은 task name, target shape, raw model output 또는 loss detail을 직접 해석하지 않는다. `ToyAnomalyAdapter`도 같은 contract를 구현하는 현재 예시일 뿐 production anomaly algorithm의 근거는 아니다.

근거:

- `cv_boilerplate@65d5412b:src/core/engine.py#Trainer._train_epoch`
- `cv_boilerplate@65d5412b:src/core/engine.py#Trainer.evaluate`
- `cv_boilerplate@65d5412b:src/tasks/toy/adapter.py#ToyAnomalyAdapter`

`권고` anomaly adapter에는 최소한 다음 책임을 둔다.

- train/eval batch에서 image, mask, label, sample identity와 auxiliary batch를 해석한다.
- upstream model의 loss/update rule을 호출해 engine이 최적화할 단일 scalar loss를 만든다.
- raw output에서 image-level score, pixel-level map 및 필요 metadata를 명시적 내부 contract로 정규화한다.
- metric update, fit 전후 calibration, model buffer 저장 필요성을 hook과 `extra_final_metrics()`로 선언한다.
- predict에는 JSON/CSV로 저장 가능한 sample별 요약을 반환하고, 큰 map 또는 image tensor는 adapter 내부 visualization/save hook에서 처리한다.

`미결정` PatchCore memory bank fitting, EfficientAD auxiliary stream 및 STFPM teacher/student state 중 어느 것이 `on_fit_start`, `on_epoch_*`, `on_fit_end` 또는 model method에 속하는지는 pinned anomalib source별 audit 없이는 확정할 수 없다. 이는 `P2-T01`, `P3-T01`, `P4-T02`의 source diff와 reference protocol에서 결정한다.

## 7. 목표 경계와 검증

```text
registry/config
  -> anomaly model package (upstream algorithm 보존)
  -> anomaly adapter (batch, loss, output, calibration contract)
  -> common Trainer (AMP, grad clip, epoch, checkpoint)
  -> metric / output adapter hooks
```

| 판정 | 목표 위치 | 검증 |
|---|---|---|
| upstream network, loss 수식, learned buffer | model package | pinned source와 checksum/diff, unit parity |
| batch 해석과 loss orchestration | anomaly adapter | contract fixture와 gradient smoke |
| optimizer, AMP, clipping, resume | common Trainer/checkpoint | resume 및 model-state round-trip |
| model-specific fit/calibration | adapter hook 또는 명시적 model method | hook ordering과 reference comparison |
| CLI model 선택 | registry/config | unknown name 및 config validation failure |

`권고` 새 abstraction은 두 모델 이상에서 같은 lifecycle 필요성이 확인되기 전에는 추가하지 않는다. 하나의 모델만 요구하는 state 또는 auxiliary input은 adapter/model package에 남긴다. 각 model inventory entry는 source, license, local asset, adapter class, lifecycle hook, expected output contract와 reference protocol을 연결해야 한다.

작성일: 2026-08-20  
상태: 세 저장소 정적 비교 초안
