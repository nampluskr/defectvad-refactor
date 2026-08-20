# `defectvad`, `roi-corner-detection-ver3`, `cv_boilerplate`의 실행 lifecycle 비교

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

이 문서는 train, validation, evaluate, predict가 어떤 진입점에서 조립되고 model state가 어느 시점에 선택·저장·복원되는지 비교한다. 목적은 anomaly model의 fit, memory-bank 구성, calibration 같은 차이를 integration 계층에서 수용하면서 공통 lifecycle은 `cv_boilerplate`가 소유하게 하는 것이다.

Model/adapter contract는 [03_MODEL_AND_ADAPTER.md](03_MODEL_AND_ADAPTER.md), output 및 metric 의미는 [05_OUTPUT_AND_VISUALIZATION.md](05_OUTPUT_AND_VISUALIZATION.md), CLI 및 반복 subprocess 운용은 후속 06 문서에서 다룬다.

## 2. 분석 기준

분석 기준일은 2026-08-20이다. 정적 코드만 조사했으며 실제 train/evaluate/predict 또는 checkpoint resume을 실행하지 않았다.

| 저장소 | Revision | lifecycle 비교에서의 역할 |
|---|---|---|
| `defectvad` | `14879ea2a8970cee25438500e5abfeeb4be8e358` | model별 trainer 및 별도 실행 script의 레거시 근거 |
| `roi-corner-detection-ver3` | `8ae989a88996441e44fb2d5296a6419a8f661220` | 사용자가 train/evaluate/predict를 같은 wrapper로 운용하는 근거 |
| `cv_boilerplate` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` | 공통 Trainer, command 및 checkpoint lifecycle의 현재 근거 |

연결 요구사항은 `FR-001`, `FR-005`, `FR-018`~`FR-020`, `FR-022`, `FR-024`, `FR-025`, `NFR-001`, `NFR-002`, `NFR-008`, `NFR-013`, `CON-003`~`CON-005`, `CON-009`, `CON-010`이며, 관련 SPEC은 §7, §9~§12, §15, §17.3~§17.4다.

## 3. 한눈에 보는 결론

```text
defectvad
train.py -> BaseTrainer.fit(train, optional test-as-valid) -> weight 저장
evaluate.py / predict.py -> model 재생성 -> weight load -> 별도 evaluator / visualizer

roi-corner-detection-ver3
train.py -> Wrapper + Trainer.fit -> history/model 저장
evaluate.py -> Wrapper load -> Evaluator -> metrics.json
predict.py -> Wrapper load -> Predictor -> predictions.csv

cv_boilerplate
train command -> common Trainer.fit -> valid-selected best/last checkpoint
evaluate/predict command -> 동일 builder -> checkpoint load -> adapter hook
```

| 질문 | `defectvad` | `roi-corner-detection-ver3` | `cv_boilerplate` |
|---|---|---|---|
| train loop 소유 | `BaseTrainer`와 model trainer | `Trainer`와 wrapper | common `Trainer`와 adapter |
| validation source | train script가 test loader를 전달 가능 | 별도 valid split | train command가 valid split으로 조립 |
| best model 선택 | 확인된 공통 best-checkpoint protocol 없음 | early-stop path는 memory의 best state 복원 | monitor metric으로 `best.pth`, 항상 `last.pth` |
| evaluate/predict | 각 script가 model/dataset을 다시 조립 | 같은 wrapper factory를 재사용 | 같은 builder/adapter와 checkpoint loader 재사용 |
| model-specific lifecycle | trainer subclass override | wrapper warm-up hook | adapter fit/epoch hooks |
| resume provenance | model weight 중심 | output history와 model weight | optimizer/scheduler/scaler/RNG/config/env를 checkpoint에 포함 |

`권고` 공통 engine은 epoch, AMP, gradient clipping, monitor, checkpoint, resume과 split access를 소유한다. anomaly-specific fitting과 calibration은 adapter hook 또는 adapter가 호출하는 model method로 한정한다. evaluate/predict는 train 이후의 in-memory object를 믿지 않고, resolved config와 selected checkpoint를 새 process에서 다시 조립해 검증해야 한다.

## 4. `defectvad`: script별 조립과 model별 trainer

`확인된 사실` `experiments/train.py#train`은 config 병합 후 model, trainer, train/test dataset 및 loader를 생성한다. `validate` 옵션이 설정되면 test loader를 `BaseTrainer.fit(..., valid_loader=test_loader)`에 전달한다. 학습 뒤에는 선택적으로 model weight와 config를 experiment directory에 저장하고, evaluator로 image/pixel 평가를 수행한다.

근거: `defectvad@14879ea2:experiments/train.py#train`

`확인된 사실` `BaseTrainer.fit`은 model별 `configure_optimizers()`, hook, `training_step()`과 internal validation loop를 호출한다. validation loop는 내장 `Evaluator.evaluate_image_level()`에 연결된다. trainer의 scheduler step 시점과 training state는 subclass 및 base implementation에 분산된다.

근거: `defectvad@14879ea2:src/defectvad/common/base_trainer.py#BaseTrainer`

`확인된 사실` `experiments/evaluate.py`와 `experiments/predict.py`는 모두 experiment config를 다시 읽고 model/dataset/loader를 새로 만든 뒤 `BaseModel.load()`를 호출한다. predict는 prediction 결과에 대해 optional train-loader calibration과 `Visualizer`를 수행한다.

근거:

- `defectvad@14879ea2:experiments/evaluate.py#evaluate`
- `defectvad@14879ea2:experiments/predict.py#predict`
- `defectvad@14879ea2:src/defectvad/common/base_model.py#BaseModel.load`

`해석` train/test 접근과 evaluation은 한 흐름에서 따라가기 쉽지만 validation에 test가 사용될 수 있어 target selection과 최종 test 보고의 경계가 약하다. 저장 artifact가 model state 중심이므로 optimizer, scheduler, AMP scaler, RNG 및 monitor provenance를 복원하는 general resume protocol은 확인되지 않았다.

`이전 판정` script가 새 process에서 config와 weight를 다시 조립하는 원칙은 `재사용`한다. test-as-validation, evaluator가 고정된 validation, model weight만 저장하는 공통 lifecycle은 `대체`한다. 개별 anomalib algorithm의 fit 순서가 필요하면 source audit 후 adapter/model method로 `조정`한다.

## 5. `roi-corner-detection-ver3`: 공통 wrapper의 명시적 3개 명령

`확인된 사실` `src/core/trainer.py#Trainer`는 wrapper의 `train_step`, `eval_step`, `on_fit_start`, `on_epoch_start`, `on_epoch_end`를 호출한다. 일반 `fit()`은 train/optional valid history를 누적하고, `fit_early_stop()`은 monitor가 개선된 model state의 CPU copy를 메모리에 보관했다가 종료 후 복원한다. `save()`는 `history.json`을 저장한다.

근거: `roi-corner-detection-ver3@8ae989a8:src/core/trainer.py#Trainer`

`확인된 사실` wrapper는 warm-up epoch에 backbone trainability, optimizer 및 scheduler를 바꾸는 hook을 제공한다. train step 내부에서 backward와 optimizer step을 실행하며, trainer는 wrapper가 반환한 scalar들을 history로 수집한다.

근거: `roi-corner-detection-ver3@8ae989a8:src/models/base/wrapper.py#BaseWrapper`

`확인된 사실` evaluate command는 wrapper를 만들고 stored model을 load한 뒤 `Evaluator.evaluate()`와 `Evaluator.save()`를 호출한다. predict command는 `Predictor.predict()`와 `Predictor.save()`를 호출한다. 두 경로는 train과 별도 process이지만 같은 factory/wrapper 계층을 사용한다.

근거:

- `roi-corner-detection-ver3@8ae989a8:scripts/evaluate.py#main`
- `roi-corner-detection-ver3@8ae989a8:scripts/predict.py#main`
- `roi-corner-detection-ver3@8ae989a8:src/core/evaluator.py#Evaluator`
- `roi-corner-detection-ver3@8ae989a8:src/core/predictor.py#Predictor`

`해석` 명령 간 wrapper factory 재사용과 history/metrics/prediction artifact 분리는 목표 workflow에 참고할 만하다. 그러나 early-stop best state가 process memory에만 있고 wrapper가 optimizer step까지 수행하므로 checkpoint/resume 및 device/AMP policy를 여러 model에 균일하게 적용하는 공통 engine의 근거는 되지 않는다.

## 6. `cv_boilerplate`: selected checkpoint를 중심으로 한 공통 lifecycle

`확인된 사실` `Trainer.fit`은 adapter fit start hook 이후 각 epoch에 train, scheduler step, valid evaluation, epoch end hook 순서로 진행한다. configured monitor metric의 개선 시 `best.pth`를, 설정에 따라 매 epoch `last.pth`를 저장한다. fit 종료 전 best weight를 다시 load한 후 `adapter.on_fit_end()`를 호출하며, hook이 model buffer를 변경한 경우 best checkpoint의 `model_state`를 갱신한다.

근거: `cv_boilerplate@65d5412b:src/core/engine.py#Trainer.fit`

`확인된 사실` engine은 `adapter.train_step()`이 반환한 grad-connected scalar loss만 backward/optimizer/AMP scaler/gradient clipping에 사용한다. evaluate와 predict는 `torch.no_grad()`에서 adapter contract만 호출한다.

근거:

- `cv_boilerplate@65d5412b:src/core/engine.py#Trainer._train_epoch`
- `cv_boilerplate@65d5412b:src/core/engine.py#Trainer.evaluate`
- `cv_boilerplate@65d5412b:src/core/engine.py#Trainer.predict`

`확인된 사실` checkpoint에는 model, optimizer, scheduler, scaler, epoch, best metric, monitor, resolved config, environment 정보와 Python/NumPy/Torch/CUDA RNG state가 저장된다. load는 선택적으로 runtime state와 RNG를 복원한다. CLI train은 resume checkpoint를 전달하고, evaluate/predict는 checkpoint model state만 load한다.

근거:

- `cv_boilerplate@65d5412b:src/core/checkpoint.py#save_checkpoint`
- `cv_boilerplate@65d5412b:src/core/checkpoint.py#load_checkpoint`
- `cv_boilerplate@65d5412b:src/cli/commands.py#train`
- `cv_boilerplate@65d5412b:src/cli/commands.py#evaluate`
- `cv_boilerplate@65d5412b:src/cli/commands.py#predict`

`해석` selected best weight before `on_fit_end` ordering은 validation-only calibration을 checkpoint에 보존하기 위한 명시적 경계다. Anomaly model의 threshold, quantile, memory-bank 또는 teacher state가 fit 종료에서 결정되는 경우에 적합할 수 있지만, 실제로 필요한 buffer와 data split은 pinned anomalib model별로 검증해야 한다.

## 7. 목표 lifecycle과 model별 예외 처리

```text
train split -> adapter.on_fit_start
            -> epoch: train_step / common optimization / valid eval
            -> valid-selected best checkpoint
            -> adapter.on_fit_end using train + valid only
            -> calibrated selected model persisted

evaluate or predict (new process)
            -> same resolved config + selected checkpoint
            -> adapter evaluation / prediction contract
            -> immutable result + provenance
```

| lifecycle 차이 | 허용 위치 | 공통 engine에 두지 않을 것 | 검증 |
|---|---|---|---|
| auxiliary data 준비 | adapter `on_fit_start` 또는 model method | model-name branch | asset preflight와 missing-asset failure |
| feature statistics/memory bank fitting | adapter fit hook 또는 explicit `fit_*` | trainer subclass 추가 | train-only split and checkpoint reload |
| calibration/threshold 결정 | `on_fit_end`, valid-only input | predict-time silent refit | checkpoint round-trip and leakage test |
| multi-phase optimizer | adapter-declared protocol | task-specific optimizer logic in engine | phase transition fixture |
| no-gradient fitting model | adapter/model method | fake optimizer/loss | reference protocol parity |

`미결정` 모든 anomaly model이 valid split을 필요로 하는지, unsupervised reference protocol에서 train-only fitting 후 test-only reporting을 허용할지, best monitor metric을 무엇으로 고정할지는 `P0-T02`의 승인된 protocol 없이는 확정할 수 없다. 어떤 선택이든 test split으로 checkpoint 또는 threshold를 선택하지 않는 `CON-009`, `CON-010` 경계는 유지해야 한다.

작성일: 2026-08-20  
상태: 세 저장소 정적 비교 초안
