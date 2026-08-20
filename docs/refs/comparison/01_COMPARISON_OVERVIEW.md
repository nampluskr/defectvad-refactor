# `defectvad`, `roi-corner-detection-ver3`, `cv_boilerplate` 구조 비교 개요

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

이 문서는 사용자가 bottom-up 방식으로 작성한 `defectvad`, 사용자가 다양한 조건을 직접 운용하는 `roi-corner-detection-ver3`, AI 에이전트가 작성한 `cv_boilerplate`의 구조적 대응을 설명한다.

여기서 "변경"과 "일반화"는 두 저장소의 구조적 대응을 뜻한다. `defectvad`가 실제로 `cv_boilerplate`로 순차 리팩터링되었다는 역사적 계보를 뜻하지 않는다. 현재 `cv_boilerplate`의 STFPM 구현도 source 주석에서 `defectvad`의 직접 port가 아니라고 명시한다.

최종 목표는 사용자가 AI 에이전트에 의존하지 않고 다음 작업을 직접 수행하도록 돕는 것이다.

- dataset과 model을 추가하거나 교체한다.
- transform, batch size, optimizer, metric과 평가 조건을 변경한다.
- model별 lifecycle 차이를 적절한 extension point에 구현한다.
- 같은 구성을 CLI 또는 notebook에서 조립해 실행한다.
- 같은 조건을 batch config로 반복 실행하고 실패와 결과를 비교한다.
- 가용하다고 판정된 모든 SOTA anomaly model의 source, license, asset, lifecycle과 reference protocol을 inventory로 관리한다.
- 공통 engine과 model 또는 adapter 중 어디를 변경해야 하는지 구분한다.

이 문서는 전체 지도이며 세부 구현은 [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md) 이후 문서에서 다룬다.

## 2. 분석 기준

분석 기준일은 2026-08-20이다.

이 문서는 `defectvad`, `roi-corner-detection-ver3@8ae989a88996441e44fb2d5296a6419a8f661220`, `cv_boilerplate` 세 저장소를 비교한다. ROI 프로젝트는 anomaly architecture의 기준이 아니라 사용자가 model, network, head, dataset과 실행 조건을 직접 조립하고 CLI와 batch를 운용하는 사용성 근거다. 최종 목적은 `defectvad`를 `cv_boilerplate` lifecycle에 통합하고 가용하다고 판정된 모든 SOTA anomaly model을 지속적으로 포함하는 것이다.

| 구분 | 저장소 | Branch | Revision | 상태 |
|---|---|---|---|---|
| 사용자 bottom-up 구현 | `D:\_clones\defectvad` | `main` | `14879ea2a8970cee25438500e5abfeeb4be8e358` | clean |
| 사용자 운용 workflow | `D:\_clones\roi-corner-detection-ver3` | `main` | `8ae989a88996441e44fb2d5296a6419a8f661220` | clean |
| AI 에이전트 작성 boilerplate | `D:\_clones\cv_boilerplate` | `main` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` | clean |

문장은 `확인된 사실`, `해석`, `권고`, `사용자 작업 지점`, `미결정`으로 구분한다. 기존 v0.1 문서의 공개 분석 기준은 `71261cef`이지만 이 문서는 현재 checkout `65d5412b`를 조사했다. 두 revision의 전체 diff는 아직 대조하지 않았다.

## 3. 한눈에 보는 결론

`확인된 사실` `defectvad`는 anomaly model 실행이라는 구체적인 workflow 안에서 공통 기능을 추출했다. `roi-corner-detection-ver3`는 model, network, head, dataset을 사용자가 직접 조합하고 같은 조건으로 train, evaluate, predict와 batch를 실행한다. `cv_boilerplate`는 여러 vision task와 model을 같은 engine에서 실행하기 위해 생성, task semantics, lifecycle과 orchestration 경계를 나눴다.

```text
defectvad
실행 스크립트
  -> config 병합
  -> module/class 문자열로 객체 생성
  -> model별 BaseModel 및 BaseTrainer
  -> 별도 Evaluator와 Visualizer

roi-corner-detection-ver3
공통 CLI 또는 Python batch config
  -> model + network + head + dataset 조립
  -> 명시적 factory로 wrapper 생성
  -> train/evaluate/predict script
  -> subprocess별 실패 격리와 산출물

cv_boilerplate
CLI 또는 notebook
  -> config resolve 및 validation
  -> registry 이름으로 구성요소 조회
  -> builder로 공통 객체 생성
  -> model + TaskAdapter 조립
  -> 공통 Trainer의 fit/evaluate/predict
```

| 질문 | `defectvad` | `roi-corner-detection-ver3` | `cv_boilerplate` |
|---|---|---|---|
| 구현 선택 | YAML의 `module`과 `class` | model/network/head/dataset option과 명시적 factory | registry의 논리 이름 |
| batch 의미 | wrapper, trainer와 evaluator가 직접 가정 | wrapper가 final corner contract로 정규화 | `TaskAdapter`가 제공 |
| 학습 loop | `BaseTrainer`와 model별 trainer | 공통 trainer와 model wrapper | 공통 `Trainer`, model `train_step`, adapter hook |
| 평가와 예측 | `Evaluator`, `BaseModel.predict`, 실행 스크립트 | 별도 CLI가 공통 factory와 wrapper 재사용 | `Trainer.evaluate/predict`와 adapter |
| 조건 변경 | Python 상수와 병합 YAML | 공통 CLI option과 Python `CONFIGS` | 상속 YAML과 CLI `--set` |
| 반복 실험 | Python 목록과 subprocess | config별 mode subprocess와 실패 격리 | benchmark YAML과 runner |

`해석` `defectvad`의 공통화 단위는 anomaly model 사이에서 반복되는 코드다. ROI 프로젝트는 사용자가 조립할 선택 축과 반복 실행 조건을 명시한다. `cv_boilerplate`의 공통화 단위는 task와 model이 달라도 유지할 수 있는 실행 protocol이므로 목표 workflow는 ROI의 운용성을 이 protocol 위에 표현해야 한다.

## 4. `defectvad`의 bottom-up 구조

### 4.1 실행 스크립트가 workflow를 조립한다

`확인된 사실` `experiments/train.py`는 argument parsing, config 병합, 출력 경로, seed, logging, 객체 생성, 학습, 평가와 저장을 한 흐름에서 조립한다.

```python
def train(config):
    anomaly_model = create_model(config["model"]).info()
    trainer = create_trainer(anomaly_model, config["trainer"])

    train_dataset = create_dataset("train", config["dataset"])
    test_dataset = create_dataset("test", config["dataset"])
    train_loader = create_dataloader(train_dataset, config["train_loader"])
    test_loader = create_dataloader(test_dataset, config["test_loader"])

    trainer.fit(train_loader, valid_loader=test_loader, ...)
    Evaluator(anomaly_model).evaluate_image_level(test_loader)
```

근거: `defectvad@14879ea2:experiments/train.py#train`

`해석` 사용자는 실행 순서를 위에서 아래로 쉽게 추적할 수 있다. 반면 train, evaluate와 predict 스크립트가 config, dataset, dataloader와 model 생성 절차를 각각 조립해 workflow가 늘어날수록 책임이 여러 진입점에 반복된다.

### 4.2 factory가 Python 경로를 해석한다

`확인된 사실` factory는 YAML의 module 경로와 class 이름을 `importlib`과 `getattr`로 해석한다.

```python
def create_model(config):
    module = importlib.import_module(config["module"])
    module_class = getattr(module, config["class"])
    if config["params"] is not None:
        return module_class(**config["params"])
    return module_class()
```

근거: `defectvad@14879ea2:src/defectvad/common/factory.py#create_model`

STFPM config는 model과 trainer의 경로를 직접 기록한다.

```yaml
model:
  module: defectvad.models.stfpm.model_trainer
  class: STFPM
trainer:
  module: defectvad.models.stfpm.model_trainer
  class: STFPMTrainer
```

근거: `defectvad@14879ea2:configs/models/stfpm.yaml`

`해석` 새 class를 config에서 바로 선택할 수 있지만 config가 package 구조를 알아야 한다. 경로 오타나 contract 불일치는 import 또는 실행 시점에 드러난다.

### 4.3 wrapper, trainer와 evaluator가 anomaly 의미를 안다

`확인된 사실` `BaseModel`은 raw model, device 이동, 입력 유형별 prediction, dataloader 결과 병합과 weight save/load를 함께 담당한다.

```python
class BaseModel:
    def predict(self, data):
        if isinstance(data, torch.Tensor): ...
        if isinstance(data, dict): ...
        if isinstance(data, DataLoader): ...

    def save(self, weights_path): ...
    def load(self, weights_path, strict=True): ...
```

근거: `defectvad@14879ea2:src/defectvad/common/base_model.py#BaseModel`

`BaseTrainer`는 optimizer 구성, hook, epoch/batch 상태, train loop와 validation loop를 제공한다. model별 trainer가 `training_step()`을 구현한다. `Evaluator`는 `pred_score`와 `anomaly_map`을 직접 읽어 image/pixel metric과 threshold를 계산한다.

근거:

- `defectvad@14879ea2:src/defectvad/common/base_trainer.py#BaseTrainer`
- `defectvad@14879ea2:src/defectvad/common/evaluator.py#Evaluator`
- `defectvad@14879ea2:src/defectvad/models/stfpm/model_trainer.py#STFPMTrainer`

`해석` anomaly model 사이의 반복은 줄였지만 공통 base class가 anomaly batch와 output을 안다. 다른 vision task로 확장하려면 이 가정을 다시 분리해야 한다.

### 4.4 반복 조건도 Python code에 있다

`확인된 사실` `experiments/run_training.py`는 dataset, category, model 목록과 `MAX_EPOCHS` 등을 Python 상수로 정의하고 조합별 `train.py` subprocess를 실행한다.

근거: `defectvad@14879ea2:experiments/run_training.py#run`

`해석` 수정 위치는 직접적이지만 비교 조건, 허용 예외, 결과 집계와 재실행 정책을 독립된 schema로 검증하기 어렵다.

## 5. `roi-corner-detection-ver3`의 사용자 운용 구조

### 5.1 선택 축, CLI와 batch orchestration

`확인된 사실` `roi-corner-detection-ver3@8ae989a8`의 canonical README와 `scripts/config.py`는 model, network, head와 dataset을 독립된 CLI 선택 축으로 정의한다. model은 target, loss와 postprocess package를, network는 backbone 또는 external whole architecture를, head는 output variant를 선택한다. reg, offset, seg, det, peak, ridge, gcn, hybrid, torchseg, torchdet, yolo와 detr wrapper는 서로 다른 raw output을 공통 normalized corner contract로 맞춘다.

근거:

- roi-corner-detection-ver3@8ae989a8:README.md §1, §3, §9
- roi-corner-detection-ver3@8ae989a8:scripts/config.py#DEFAULTS
- roi-corner-detection-ver3@8ae989a8:scripts/config.py#get_wrapper_kwargs
- roi-corner-detection-ver3@8ae989a8:src/core/factory.py#get_wrapper

`확인된 사실` `train.py`, `evaluate.py`와 `predict.py`는 같은 parser, dataset factory와 wrapper factory를 사용한다. train은 `history.json`과 선택적 `model.pth`를, evaluate는 `metrics.json`을, predict는 `predictions.csv`를 실험 `output_dir`에 저장한다. `scripts/batch_config.py`는 `BASE`에 model, network, head와 실행 조건을 덮어쓴 `CONFIGS` 목록을 만들고, `scripts/batch_run.py`는 train, evaluate, predict 또는 all mode를 subprocess로 실행한다.

근거:

- roi-corner-detection-ver3@8ae989a8:scripts/train.py#main
- roi-corner-detection-ver3@8ae989a8:scripts/evaluate.py#main
- roi-corner-detection-ver3@8ae989a8:scripts/predict.py#main
- roi-corner-detection-ver3@8ae989a8:scripts/batch_config.py#CONFIGS
- roi-corner-detection-ver3@8ae989a8:scripts/batch_run.py#get_cli_args
- roi-corner-detection-ver3@8ae989a8:scripts/batch_run.py#run

`확인된 사실` batch runner는 한 subprocess 실패를 success/error 결과로 남기고 다음 조합을 계속 실행한 뒤, 실패가 하나라도 있으면 최종 exit code 1을 반환한다. 다만 결과 요약을 machine-readable artifact로 저장하지 않고 `PASS_KEYS`가 공통 parser의 모든 option을 전달하지 않는다. automatic experiment name도 전체 data와 training identity를 포함하지 않아 `output_dir`를 명시하지 않으면 덮어쓸 수 있다.

`해석` ROI 프로젝트의 참고 가치는 사용자가 조합 축과 CLI를 바로 읽고 같은 조건으로 train, evaluate, predict를 반복할 수 있다는 점이다. `defectvad`의 Python 목록과 subprocess보다 축과 산출물 규칙이 분명하지만, `get_wrapper`의 model 이름 조건문과 Python config 주석 해제 방식은 SOTA anomaly inventory를 지속 확장할 공통 core에 그대로 옮길 architecture가 아니다.

`권고` 목표 anomaly workflow는 ROI 프로젝트의 명시적 model/dataset/실행 조건, 조합별 실패 격리와 결과 비교 경험을 `cv_boilerplate`의 registry, config override와 benchmark runner 위에 표현한다. 각 case는 source revision, license, local asset, lifecycle, reference protocol, resolved config, exit status와 artifact path를 manifest에 남겨야 한다. 공통 engine에는 anomaly task명이나 model명 분기를 추가하지 않는다.

`미결정` benchmark case가 train, evaluate, predict를 항상 연속 수행할지, reference checkpoint를 사용하는 evaluate-only case도 같은 inventory에서 표현할지는 후속 06 문서와 SPEC §11에서 결정해야 한다.

## 6. `cv_boilerplate`의 조립 구조

### 6.1 공통 진입점

`확인된 사실` `python -m src`는 task module보다 먼저 offline guard를 켠 뒤 `src.tasks`를 import해 registry를 채운다. 공통 parser와 `dispatch()`는 train, evaluate, predict, benchmark 등으로 분기한다.

```python
enable_offline_guard()
import src.tasks

def dispatch(args):
    if args.command == "train": train(...)
    elif args.command == "evaluate": evaluate(...)
    elif args.command == "predict": predict(...)
    elif args.command == "benchmark": run_benchmark(...)
```

근거: `cv_boilerplate@65d5412b:src/__main__.py#dispatch`

각 command는 config resolve와 validation 이후 transform, model, loss, metric, adapter, dataset, dataloader, optimizer와 scheduler를 조립한다.

근거:

- `cv_boilerplate@65d5412b:src/cli/commands.py#train`
- `cv_boilerplate@65d5412b:src/cli/commands.py#evaluate`
- `cv_boilerplate@65d5412b:src/cli/commands.py#predict`

### 6.2 registry가 역할 이름을 제공한다

`확인된 사실` `Registry`는 namespace별 이름과 class/function 대응을 보관하고 중복과 미등록 이름을 `RegistryError`로 보고한다.

```python
class Registry:
    def register(self, name): ...
    def get(self, name): ...
    def build(self, name, *args, **params): ...

DATASETS = Registry("dataset")
MODELS = Registry("model")
ADAPTERS = Registry("adapter")
```

근거: `cv_boilerplate@65d5412b:src/core/registry.py#Registry`

구성요소는 decorator로 등록하고 config는 논리 이름을 참조한다.

```python
@MODELS.register("stfpm_anomaly")
class Stfpm(nn.Module): ...
```

```yaml
model:
  name: stfpm_anomaly
```

근거:

- `cv_boilerplate@65d5412b:src/tasks/anomaly/models/stfpm.py#Stfpm`
- `cv_boilerplate@65d5412b:configs/anomaly/stfpm.yaml`

`해석` config에서 Python module 경로를 제거하고 역할별 catalog를 만들어 이름 validation이 가능해졌다. 대신 등록 module이 import되어야 하므로 새 파일 작성 후 package `__init__.py` 연결도 필요하다.

### 6.3 `TaskAdapter`가 task 의미를 격리한다

`확인된 사실` 공통 `Trainer`는 batch 구조나 metric 이름을 직접 해석하지 않는다. `TaskAdapter`를 통해 step, metric, prediction, batch size, collate와 lifecycle hook을 호출한다.

```python
class TaskAdapter:
    def train_step(self, model, batch, device): ...
    def eval_step(self, model, batch, device): ...
    def update_metrics(self, outputs): ...
    def predict_step(self, model, batch, device): ...
    def collate_fn(self): ...
    def on_fit_start(self, model, loaders, device): ...
    def on_fit_end(self, model, loaders, device): ...
```

근거: `cv_boilerplate@65d5412b:src/core/adapter.py#TaskAdapter`

`AnomalyAdapter`는 anomaly batch와 output, metric, threshold, smoothing, prediction과 visualization 연결을 담당한다. STFPM model은 feature extraction, training loss와 anomaly map 계산을 담당한다.

근거:

- `cv_boilerplate@65d5412b:src/tasks/anomaly/adapter.py#AnomalyAdapter`
- `cv_boilerplate@65d5412b:src/tasks/anomaly/models/stfpm.py#Stfpm`

책임 대응은 다음과 같다.

```text
algorithm과 model state      -> task model
batch와 output 의미          -> TaskAdapter
epoch/batch 실행 순서        -> 공통 Trainer
optimizer와 DataLoader 생성  -> builders
checkpoint 전체 상태         -> checkpoint module
CLI와 결과 디렉터리 조립     -> commands와 RunContext
```

### 6.4 공통 `Trainer`

`확인된 사실` `Trainer`는 `fit()`, `evaluate()`, `predict()`를 제공한다. `fit()`은 adapter hook, train epoch, validation, best/last checkpoint와 fit 종료 후 state 반영을 공통 순서로 실행한다.

근거: `cv_boilerplate@65d5412b:src/core/engine.py#Trainer`

`해석` `BaseTrainer`, `Evaluator`, `BaseModel.predict_dataloader()`에 분산된 반복 제어가 공통 engine에 모였다. model별 차이가 사라진 것은 아니며 model의 `train_step()`과 adapter hook으로 이동했다.

### 6.5 config와 benchmark

`확인된 사실` anomaly base config는 runtime, data, loss, metrics, adapter, optimizer, scheduler, train과 output을 하나의 schema로 표현한다. model config는 `_base`를 상속하고 CLI `--set`으로 leaf 값을 override한다.

근거:

- `cv_boilerplate@65d5412b:configs/anomaly/_base.yaml`
- `cv_boilerplate@65d5412b:configs/anomaly/stfpm.yaml`
- `cv_boilerplate@65d5412b:src/core/config.py#resolve_config`
- `cv_boilerplate@65d5412b:src/cli/parser.py#build_parser`

benchmark runner는 benchmark YAML의 split config를 resolve하고 split별 실행과 결과 집계를 수행한다.

근거: `cv_boilerplate@65d5412b:src/bench/runner.py#run_benchmark`

`해석` Python 상수와 subprocess loop로 표현하던 반복 조건이 config data로 이동했다. 실행 code를 덜 바꾸는 대신 사용자는 config schema와 registry 이름을 이해해야 한다.

## 7. 구성요소별 책임 이동

| 구성요소 | `defectvad` | `roi-corner-detection-ver3` | `cv_boilerplate` | 핵심 변화 |
|---|---|---|---|---|
| dataset | `data/*.py`, factory | CSV, stage, 공통 factory | task dataset, `DATASETS` | 경로와 stage를 검증 가능한 data contract로 이동 |
| transform | dataset factory 내부 | split별 공통 factory | task transform, `TRANSFORMS` | dataset 생성과 transform 선택 분리 |
| dataloader | 단순 factory | 공통 factory와 subset option | 공통 builder | seed, collate와 split 보호 포함 |
| model | `torch_model.py`, `model_trainer.py` | model/network/head wrapper | task models, `MODELS` | algorithm과 integration 책임 분리 |
| wrapper/adapter | `BaseModel` | 공통 final corner wrapper | `TaskAdapter` | task contract를 engine에서 격리 |
| trainer | `BaseTrainer`, model별 trainer | 공통 `Trainer`와 wrapper | 공통 `Trainer`, model step, adapter hook | loop와 variation 분리 |
| evaluator/predictor | 별도 class와 script | 별도 CLI, 공통 wrapper | `Trainer.evaluate/predict`, adapter | 명령은 통일하고 의미는 adapter가 소유 |
| CLI | 실행별 parser | 공통 parser의 명시적 선택 축 | 공통 parser와 dispatch | 동일 조건을 공통 config로 표현 |
| batch process | Python 목록과 subprocess | Python `CONFIGS`, mode별 subprocess | benchmark config와 runner | 가독성과 실패 격리를 manifest 기반 실행으로 이동 |
| 결과 | model별 결과 | history, metrics, predictions, weight | run context와 benchmark 결과 | provenance와 artifact identity 추가 필요 |

이 표는 파일의 일대일 이동이 아니라 책임의 분리와 결합 관계를 나타낸다.

## 8. 신규 방식의 역할

도입 의사결정 기록을 모두 확인하지 않았으므로 아래의 이유는 구조적 효과를 바탕으로 한 `해석`이다.

| 방식 | 해결하려는 문제에 대한 해석 | 적용 방식 | 제약 |
|---|---|---|---|
| registry | config와 package 경로 결합, 사전 검증 부족 | namespace별 register/get/build | import되어야 등록되고 이름은 유일해야 함 |
| `TaskAdapter` | engine이 task별 batch/output을 아는 문제 | engine은 abstract contract만 호출 | task semantics를 engine 분기로 넣지 않음 |
| builder | 생성 규칙 반복 | config spec과 registry로 생성 | model별 분기를 공통 builder에 넣지 않음 |
| config validation | 잘못된 조건의 늦은 실패 | resolve 후 registry/path/runtime 검사 | field 추가 시 validation 영향 확인 |
| `RunContext` | seed, device, AMP, provenance 분산 | resolved config로 context 생성 | notebook도 같은 초기화 순서 필요 |
| checkpoint | weight만으로 전체 상태 복원 불가 | optimizer, scheduler, RNG와 best/last 관리 | model별 비학습 state 소유권 필요 |
| offline guard | import나 weight 생성 중 network 접근 | 시작 시 차단, 명시적 opt-in | notebook은 import 순서를 직접 보장 |
| benchmark runner | 반복 조건과 결과 집계의 암묵성 | benchmark YAML과 split 실행 | 단일 run config와 benchmark 조건 구분 |

## 9. 사용자가 직접 변경하는 지도

### 9.1 조건만 변경

제품 code보다 config를 먼저 바꾼다.

| 목적 | 우선 변경 지점 | code 변경 |
|---|---|---|
| batch size, epoch, learning rate | config 또는 `--set` | 일반적으로 불필요 |
| dataset root, category, split | `data` config | 일반적으로 불필요 |
| metric 선택 | `metrics` config | 등록된 metric이면 불필요 |
| model 선택 | `model.name` | 등록된 model이면 불필요 |
| smoothing 등 adapter parameter | `adapter.params` | 지원 parameter면 불필요 |
| 반복 비교 | benchmark YAML | 일반적으로 불필요 |
| model/dataset 조건 조합 | benchmark case config | runner code 변경 불필요 |
| SOTA model inventory | model inventory 및 case metadata | 새 model integration 때 갱신 |

```text
python -m src train configs/anomaly/stfpm.yaml --set train.epochs=10 --set data.batch_size=4
```

이 명령은 parser와 override 구현을 근거로 작성했으며 실행하지 않았다.

### 9.2 새 model 추가

`사용자 작업 지점`

1. `src/tasks/anomaly/models/<model>.py`에 algorithm과 state를 구현한다.
2. `@MODELS.register("<name>")`로 등록한다.
3. `src/tasks/anomaly/models/__init__.py`에서 import한다.
4. 기존 `AnomalyAdapter` contract로 실행 가능한지 확인한다.
5. model별 fitting/state는 우선 model method와 adapter hook으로 수용한다.
6. `configs/anomaly/<model>.yaml`에서 base와 parameter를 연결한다.
7. 공통 engine 변경은 둘 이상의 model에 같은 요구가 있는지 확인한다.
8. source revision, license, local asset, lifecycle과 reference protocol을 SOTA inventory에 기록한다.

### 9.3 dataset과 평가 추가

dataset은 task dataset module, `DATASETS` 등록, import와 config를 함께 변경한다. batch 표현이 달라지면 collate와 adapter의 `batch_size`, train/eval/predict step도 검토한다.

metric은 구현과 `METRICS` 등록 후 config에서 선택한다. score/map 의미와 threshold가 달라지면 adapter와 postprocess contract도 확인한다.

### 9.4 batch config와 inventory

`사용자 작업 지점`

1. benchmark config에서 model, dataset, checkpoint, protocol과 override를 조합한다.
2. 조합별 resolved config와 output directory가 충돌하지 않는지 확인한다.
3. train, evaluate, predict 또는 evaluate-only mode를 case에 명시한다.
4. 실패한 case가 다음 case를 막지 않고 exit status와 artifact path를 manifest에 남기는지 확인한다.
5. 새 SOTA model은 가용성 판정 근거와 reference tolerance를 inventory에 추가한다.

### 9.5 engine 수정 판단

다음을 확인한 뒤 `core/engine.py`를 변경한다.

1. 하나의 model에만 필요한가.
2. model method나 adapter hook으로 표현할 수 없는가.
3. 둘 이상의 task 또는 model에 같은 순서가 필요한가.
4. 기존 classification, detection, segmentation을 깨지 않는가.
5. CLI와 benchmark가 같은 lifecycle을 유지하는가.

공통 engine에서 anomaly task명이나 model 이름으로 분기하지 않는다.

## 10. CLI, notebook과 batch

`확인된 사실` CLI는 `config`, `train`, `evaluate`, `predict`, `benchmark`, `leaderboard` command를 제공한다. 조건 변경에는 새 실행 파일보다 config와 `--set`을 우선 사용한다.

`확인된 사실` 현재 checkout에는 notebook 전용 runner가 없다. CLI 조립 함수와 core class는 Python 함수와 class이므로 notebook에서 import할 수 있다.

`확인된 사실` ROI 프로젝트의 `batch_run.py`는 Python `CONFIGS`를 train, evaluate, predict 또는 all mode로 실행하고 조합별 subprocess 실패 후에도 다음 case를 계속한다. 목표 workflow에서는 같은 사용성을 `cv_boilerplate` benchmark config와 machine-readable result manifest로 제공한다.

`권고` notebook은 CLI code를 복사해 새 lifecycle을 만들지 말고 같은 순서를 사용한다.

```text
offline guard
  -> src.tasks import로 registry 채우기
  -> resolve_config와 validate_config
  -> RunContext와 seed
  -> transform/model/adapter/dataset/dataloader build
  -> Trainer.fit/evaluate/predict
```

`미결정` 공식 programmatic API는 아직 별도로 정의되지 않았다. `src.cli.commands` 내부 함수를 직접 쓸지 안정된 helper API를 둘지는 [06_CLI_AND_BATCH_ORCHESTRATION.md](06_CLI_AND_BATCH_ORCHESTRATION.md)에서 검토한다.

## 11. 현재 주의점

### 11.1 직접 계보를 단정하지 않는다

`확인된 사실` 사용자는 두 저장소의 작성 배경을 명시했다. 현재 `cv_boilerplate` STFPM 주석은 `defectvad` 직접 port가 아니라고 밝힌다.

`해석` 따라서 "파일이 어떻게 바뀌었는가"보다 "같은 문제의 책임을 어디에 배치했는가"를 비교한다.

### 11.2 지원 model 범위는 다르다

`확인된 사실` `defectvad`에는 다수 anomaly model config가 있다. 현재 `cv_boilerplate/src/tasks/anomaly/models/__init__.py`는 `custom_ae`, `efficientad`, `stfpm`을 import하며 PatchCore는 없다.

`해석` 구조의 일반성과 model coverage는 다른 문제다. 구조 비교를 기능 동등성으로 해석하지 않는다.

### 11.3 abstraction도 학습 비용이 있다

`해석` registry와 adapter는 engine 분기를 줄이지만 사용자가 이해해야 할 연결 지점을 늘린다. 새 model은 구현뿐 아니라 등록, import, config와 adapter contract를 함께 확인해야 한다.

## 12. 후속 문서

| 알고 싶은 내용 | 문서 |
|---|---|
| dataset부터 batch까지 | [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md) |
| model, wrapper와 adapter | [03_MODEL_AND_ADAPTER.md](03_MODEL_AND_ADAPTER.md) |
| train/evaluate/predict loop | [04_EXECUTION_LIFECYCLE.md](04_EXECUTION_LIFECYCLE.md) |
| output, metric과 시각화 | [05_OUTPUT_AND_VISUALIZATION.md](05_OUTPUT_AND_VISUALIZATION.md) |
| CLI, notebook와 반복 실험 | [06_CLI_AND_BATCH_ORCHESTRATION.md](06_CLI_AND_BATCH_ORCHESTRATION.md) |
| registry와 platform mechanism | [07_PLATFORM_MECHANISMS.md](07_PLATFORM_MECHANISMS.md) |
| 이전 판정과 gap | [08_MIGRATION_SUMMARY.md](08_MIGRATION_SUMMARY.md) |
| source 근거 | [09_EVIDENCE_INDEX.md](09_EVIDENCE_INDEX.md) |

`02_DATA_PIPELINE.md`는 작성되었으며 세 저장소 기준으로 갱신되었다. `03_MODEL_AND_ADAPTER.md` 이후 링크는 예정 경로다.

## 13. 요구사항 연결

| 주제 | 요구사항 | 설계와 계획 |
|---|---|---|
| 공통 workflow | FR-001, FR-002, FR-010, FR-011 | SPEC §2, §8–§10, PLAN P1–P2 |
| model과 integration 분리 | FR-003, FR-004, NFR-003 | SPEC §6–§7, PLAN P2–P4 |
| task/model 독립 engine | FR-024, NFR-004, NFR-005, CON-005 | SPEC §3, §6–§7, PLAN P3–P4 |
| config, SOTA inventory와 반복 실험 | FR-018, FR-022, FR-024, FR-025, NFR-002 | SPEC §11, §13, §15, §17, PLAN P0, P3, P5–P6 |
| local/offline | FR-023, NFR-006, CON-006, CON-007 | SPEC §13, §16, PLAN P0, P1, P6 |

작성일: 2026-08-20  
상태: 세 저장소 비교 초안
