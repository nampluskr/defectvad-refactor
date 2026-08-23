---
name: add-anomalib-model
description: anomalib의 image 모델을 이 프로젝트로 포팅해 새 모델을 추가한다. "anomalib 모델 추가", "PatchCore/PaDiM/FastFlow 붙여줘", "새 모델 포팅", "세 번째 모델 추가"처럼 upstream anomalib 모델을 이 레포의 models/ + adapters/ + configs 구조로 이식할 때 사용한다. 데이터셋 추가에는 사용하지 않는다.
---

# anomalib 모델 추가

이 문서만 보고 새 모델 추가를 수행할 수 있도록 정리한 절차다. 기준 사례는 이미 통합된 STFPM과 EfficientAD이며, 판단이 서지 않는 지점은 그 두 모델의 실제 코드를 확인한다.

- `src/tasks/anomaly/models/stfpm/` · `src/tasks/anomaly/models/efficientad/`
- `src/tasks/anomaly/adapters/stfpm.py` · `src/tasks/anomaly/adapters/efficientad.py`
- `configs/anomaly/models/stfpm.yaml` · `configs/anomaly/models/efficientad.yaml`

## 1. 전제

- 대상 anomalib은 **`091ca6a`(tag `v2.3.0`)에 고정**되어 있다. 로컬 클론은 `/mnt/d/projects/clones/anomalib`이며 원격은 `https://github.com/open-edge-platform/anomalib.git`이다.
- 대상은 `src/anomalib/models/image/<model>/`의 image 모델로 한정한다.
- 3대 원칙은 예외 없다. 모델 코드 무수정(CON-001), Lightning 미사용(CON-002), 로컬 자산만 사용(CON-003·CON-004).
- 실행 전 conda 환경을 활성화한다.

  ```bash
  conda activate pytorch_env
  ```

### 1.1 로컬 클론 점검 (단계 0)

**모델 파일을 한 줄이라도 복사하기 전에 반드시 수행한다.** 목표는 "최신"이 아니라 **"핀에 고정되어 있고 오염되지 않았음"** 이다. 이 프로젝트의 CON-001 검증은 복사본을 특정 commit의 원본과 diff하는 것이므로, 기준이 흔들리면 검증 자체가 성립하지 않는다.

네 가지를 순서대로 확인한다. 아래 명령은 모두 읽기 전용이며, 클론을 변경하는 명령은 사용자 승인 없이 실행하지 않는다.

```bash
CLONE=/mnt/d/projects/clones/anomalib
PIN=091ca6aca92c8d0e416394f79e52f5a3cea3db73

# (1) 핀 일치
git -C $CLONE rev-parse HEAD

# (2) 작업 트리 청결
git -C $CLONE status --porcelain

# (3) 대상 모델 존재 여부
git -C $CLONE sparse-checkout list
ls $CLONE/src/anomalib/models/image/

# (4) 상류 신버전 확인 (네트워크 사용)
git -C $CLONE ls-remote --tags --refs origin | tail -5
```

판정과 조치:

| 확인 | 기대값 | 어긋날 때 |
|---|---|---|
| (1) 핀 | `$PIN`과 일치 | **중단하고 사용자에게 보고한다.** 임의로 checkout 하지 않는다. 기존 STFPM·EfficientAD가 이 핀에서 포팅되었으므로, 다른 commit에서 새 모델만 가져오면 레포 안에 서로 다른 두 기준이 섞인다 |
| (2) 트리 | 출력 없음 | 오염된 클론은 diff 검증이 무의미하다. 변경 내역을 사용자에게 보여주고 `git -C $CLONE checkout -- .` 또는 원인 확인을 요청한다 |
| (3) 모델 | 대상 디렉터리 존재 | sparse-checkout에 없으면 추가를 **요청**한다(§1.2). partial clone이라 네트워크 fetch가 발생한다 |
| (4) 태그 | 최신 태그가 `v2.3.0` | 더 새 태그가 있으면 **보고만 한다**(§1.3). pull 하지 않는다 |

### 1.2 대상 모델이 로컬에 없을 때

클론은 `blob:none` partial clone + sparse-checkout이라 `src/anomalib/models/image/` 아래에 일부 모델만 내려와 있다(초기 상태: `stfpm`, `efficient_ad`). 대상 모델이 없으면 아래를 **사용자에게 제시하고 승인을 받은 뒤** 실행한다.

```bash
git -C /mnt/d/projects/clones/anomalib sparse-checkout add src/anomalib/models/image/<model>
```

- 이 명령은 핀을 바꾸지 않는다. `091ca6a` 시점의 해당 디렉터리만 추가로 내려받는다.
- `git pull`이나 `git checkout main`으로 대체하지 않는다. 핀이 풀린다.
- 추가 후 (1)~(2)를 다시 확인한다.
- 모델이 `components/`의 새 모듈에 의존하면 그 경로도 같은 방식으로 추가한다.

### 1.3 상류에 새 버전이 있을 때

핀보다 새로운 태그를 발견하면 **작업을 계속하되**, 다음 형식으로 사용자에게 알린다.

```text
로컬 클론: 091ca6a (v2.3.0), clean
상류 최신 태그: <tag> (<commit>)
main HEAD: <commit>  # 미출시 커밋 포함
대상 모델 <model>의 변경 여부: <해당 태그에서 이 모델 디렉터리가 바뀌었는지>
```

핀 상향은 **이 작업의 범위가 아니라 별도 결정**이다. 다음을 근거로 사용자가 판단한다.

- 핀을 올리면 기존 모델의 원본도 함께 바뀌므로, STFPM·EfficientAD를 포함한 **모든 모델을 재복사·재diff·재검증**해야 한다.
- 새 모델을 신버전에서만 가져오는 절충은 허용하지 않는다. 레포 안에 기준 commit이 둘 이상 존재하게 된다.
- 사용자가 상향을 승인하면 detached HEAD 상태이므로 `git pull`이 아니라 fetch 후 태그 checkout이며, 이때 이 문서 §1의 핀 값도 함께 갱신한다.

  ```bash
  git -C /mnt/d/projects/clones/anomalib fetch --tags origin
  git -C /mnt/d/projects/clones/anomalib checkout <tag>
  ```

특정 모델이 그 사이 바뀌었는지만 보려면 다음으로 확인한다.

```bash
git -C /mnt/d/projects/clones/anomalib log --oneline $PIN..<tag> -- src/anomalib/models/image/<model>/
```

## 2. 절차 요약

| 단계 | 작업 | 산출물 |
|---|---|---|
| 0 | 로컬 클론 점검 — 핀 일치·트리 청결·모델 존재·신버전 알림 (§1.1) | 진행 가능 판정 |
| 1 | 대상 모델 파일 구성 조사 | 복사 대상 목록 |
| 2 | 의존 모듈 추적 (Lightning 오염 검사 포함) | 추가 복사 대상 |
| 3 | `models/<model>/`에 복사 + import 경로만 치환 | 신규 모델 파일 |
| 4 | `lightning_model.py` 정독 → 이관표 작성 | config 항목 / adapter hook 목록 |
| 5 | `models/<model>/__init__.py`에 no-download 팩토리 작성 | `MODELS` 엔트리 |
| 6 | `adapters/<model>.py` 작성 | `ADAPTERS` 엔트리 |
| 7 | `configs/anomaly/models/<model>.yaml` 작성 | config |
| 8 | 등록·스모크·무결성 확인 | 동작 확인 |
| 9 | 성능 검증 (사용자 실행) + 적대적 검증 + 문서 갱신 | reference 비교, `reviews/A{n}.md` |

## 3. 단계별 상세

### 3.1 단계 1 — 파일 구성 조사

§1.1의 점검을 통과한 뒤에 시작한다.

```bash
ls /mnt/d/projects/clones/anomalib/src/anomalib/models/image/<model>/
```

구성은 모델마다 다르다.

- STFPM: `torch_model.py`, `loss.py`, `anomaly_map.py`, `lightning_model.py`
- EfficientAD: `torch_model.py`, `lightning_model.py` (loss·anomaly map이 `torch_model.py` 안에 있다)

복사 대상은 **`lightning_model.py`와 `__init__.py`를 제외한 순수 PyTorch 파일 전부**다. anomalib의 패키지 `__init__.py`는 `lightning_model`을 import하므로 복사하지 않는다.

`torch_model.py`의 `forward`는 보통 학습 시 features를, 평가 시 `InferenceBatch(pred_score, anomaly_map)`를 반환한다. adapter 작성 전에 이 분기를 확인한다.

### 3.2 단계 2 — 의존 모듈 추적

복사 대상 파일의 모든 `from anomalib...` import를 추적한다. 아래 두 함정이 거의 모든 모델에서 재현된다.

1. **`components/__init__.py`를 복사하면 Lightning이 딸려 온다.** `components/base/anomalib_module.py`가 `import lightning.pytorch as pl`을 한다. 패키지 `__init__`이 아니라 **실제로 필요한 모듈 파일만** 옮기고 import는 모듈 단위로 직접 건다.
2. **`from anomalib.data import InferenceBatch`는 `anomalib.data.__init__`을 통해 Lightning `DataModule`에 닿는다.** 이미 `src/tasks/anomaly/models/components/data/torch_base.py`로 복사되어 있으므로 새 모델은 그 경로를 재사용한다.

재사용 가능한 기존 components는 다음과 같다. 필요한 것이 여기 있으면 다시 복사하지 않는다.

```text
src/tasks/anomaly/models/components/
├── data/torch_base.py                  # InferenceBatch 등
├── data/generic.py
├── feature_extractors/timm.py          # TimmFeatureExtractor
├── feature_extractors/utils.py
└── tiler.py
```

새 components가 필요하면 같은 규칙으로 **파일째** 복사한다. 필요한 클래스만 발췌하지 않는다 — 파일 단위 복사가 CON-001의 diff 검증을 가능하게 한다.

검사:

```bash
grep -rn "lightning" src/tasks/anomaly/models/
```

docstring 외에 실행 경로 언급이 없어야 한다. 기존에 남은 건은 모두 docstring이며, **지우면 CON-001 위반이다.**

### 3.3 단계 3 — 복사와 import 치환

```text
src/tasks/anomaly/models/<model>/     # anomalib 원본 — 수정 금지 구역
```

허용되는 변경은 import 경로 치환 **한 가지뿐**이다.

```python
from anomalib.data import InferenceBatch
# ->
from src.tasks.anomaly.models.components.data.torch_base import InferenceBatch
```

복사 직후 원본과 diff해 import 외 차이가 0라인인지 확인하고 기록한다.

```bash
diff -u /mnt/d/projects/clones/anomalib/src/anomalib/models/image/<model>/torch_model.py \
        src/tasks/anomaly/models/<model>/torch_model.py
```

`AGENTS.md`의 Code Style Rules는 이 디렉터리에 적용하지 않는다. 스타일 통일을 명목으로 한 수정은 금지다.

### 3.4 단계 4 — `lightning_model.py` 이관표

`lightning_model.py`는 **복사하지 않고 읽는다.** 읽으면서 아래 표를 채운다. 이 표가 단계 5~7의 작업 지시서가 된다.

| anomalib `lightning_model.py` | 이 프로젝트에서의 자리 |
|---|---|
| `configure_optimizers` | `configs/anomaly/models/<model>.yaml`의 `optim` (코드 아님) |
| `training_step`의 loss 조합 | `<Model>Adapter.train_step` |
| `validation_step` | 보통 불필요. 공통 `AnomalyAdapter.eval_step`이 흡수 |
| `on_train_start` | `<Model>Adapter.on_fit_start` |
| `on_validation_start` | `<Model>Adapter.on_validation_start` |
| `on_train_end` / 학습 후 보정 | `<Model>Adapter.on_fit_end` (`super()` 호출 필수) |
| `prepare_*` 계열의 다운로드 | **옮기지 않는다.** 로컬 경로 strict load로 대체 |
| `trainer_arguments`의 제약(batch size 등) | config에 값으로 기재 + 주석으로 근거 |
| `learning_type` | 대응물 없음. 무시 |
| post-processing / normalization | 공통 `AnomalyAdapter` + `postprocess/`로 이미 충족되는지 먼저 확인 |

optimizer 이관 주의점: `build_optimizer`(`src/core/builders.py:25`)는 `requires_grad=True`인 파라미터로 **단일 parameter group**만 만든다. anomalib이 특정 서브모듈만 optimizer에 넣는다면 그 집합이 freeze 결과와 일치하는지 확인한다. 일치하면 config만으로 충분하다. 서로 다른 lr을 갖는 복수 group이 필요하면 `core/builders.py` 변경이 필요하며, 이는 공통 코드 변경이므로 사용자 승인을 받는다.

사용 가능한 optimizer·scheduler 빌더는 `adamw`, `adam`, `sgd`, `cosine`, `step`이다. 없으면 `core/builders.py`에 **범용** 빌더로 추가한다(모델명 분기가 아니면 NFR-005 위반이 아니다).

scheduler의 `step_size`처럼 `train.epochs`에서 파생되는 값은 config 참조 문법이 없으므로 상수로 적고 **짝이 되는 키를 주석으로 명시**한다(`configs/anomaly/models/efficientad.yaml` 참조).

### 3.5 단계 5 — no-download 모델 팩토리

팩토리는 `src/tasks/anomaly/models/<model>/__init__.py`에 둔다. 여기는 이 프로젝트가 소유하는 파일이므로 Code Style Rules를 적용한다(원본 파일만 수정 금지 구역이다).

```python
from src.core.registry import MODELS
from .torch_model import <Model>Model


@MODELS.register("<model>")
@MODELS.register("<model>_anomaly")
def build_<model>(weights_path=None, **params):
    """No-download <model> factory."""
    # build -> freeze -> local weight injection
    return model
```

`models/__init__.py`에도 export를 추가한다. 이 한 줄이 빠지면 registry에 등록되지 않는다.

```python
from src.tasks.anomaly.models.<model> import <Model>Model, build_<model>
```

upstream 생성자가 pretrained를 건드리는지에 따라 두 갈래다.

| 상황 | 처리 | 예 |
|---|---|---|
| 생성자가 pretrained를 로드/다운로드한다 | 진입점 함수(`timm.create_model` 등)를 `pretrained=False`로 덮는 wrapper로 **생성자 호출 한 문장 동안만** 치환하고 `try/finally`로 원복 → 이후 로컬 `.pth` 주입 | `build_stfpm` |
| 생성자가 가중치를 건드리지 않는다 | 팩토리는 경로만 모델 속성에 보관하고, adapter의 `on_fit_start`에서 `load_local_weights(..., strict=True)` | `build_efficientad` |

이 치환은 **`models/<model>/` 밖으로 내보내지 않는다.** `core/`에 일반화된 "pretrained 차단 컨텍스트"를 만들지 않는다.

strict load 판정:

- `missing`이 하나라도 있으면 `LocalAssetError`로 즉시 실패한다.
- `unexpected`는 **모델이 실제로 갖고 있지 않은 최상위 서브모듈 이름의 키만** 허용한다. timm이 `out_indices` 밖 스테이지를 제거하므로 `fc.`/`head.` 접두사 허용 목록으로는 부족하다. 모델이 갖고 있는 이름인데도 unexpected면 구조 불일치이므로 실패시킨다.
- 어떤 경우에도 조용한 랜덤 초기화 폴백을 두지 않는다(CON-004).

frozen 파라미터 점검:

- teacher 등 학습 대상이 아닌 서브모듈은 `requires_grad=False`가 **`build_optimizer` 호출 전**에 설정되어야 optimizer param group에서 빠진다. 팩토리 시점에 설정한다.
- `nn.ParameterDict.update()`는 새 `nn.Parameter`(기본 `requires_grad=True`)로 교체하므로, 통계·분위수 버퍼는 **update 할 때마다 다시 freeze**한다.
- upstream이 `train()`을 override해 eval을 고정하지 않는 모델은, 공통 engine의 `model.train()`이 매 epoch teacher를 학습 모드로 되돌린다. 인스턴스 수준 `train` 바인딩으로 고정한다(`build_efficientad` 참조). 원본 파일은 건드리지 않는다.

### 3.6 단계 6 — `adapters/<model>.py`

```python
from src.core.registry import ADAPTERS
from .base import AnomalyAdapter


@ADAPTERS.register("<model>")
class <Model>Adapter(AnomalyAdapter):
    def train_step(self, model, batch, device):
        ...
    # override only the lifecycle hooks this model needs
```

작성 규칙:

- `AnomalyAdapter`(`adapters/base.py`)를 상속하고 **모델별 차이만 override**한다. metric·threshold·smooth·visualize는 공통 구현을 그대로 쓴다.
- `train_step`은 `{"loss": Tensor, "loss_dict": {str: float}}`를 반환한다. loss 계산은 복사해 온 원본 모듈을 호출한다. loss를 새로 작성하지 않는다.
- `eval_step`/`predict_step`은 보통 override하지 않는다. 모델이 `InferenceBatch`를 반환해도 공통 구현이 `(B,)`/`(B,H,W)`로 변환한다.
- **hook을 override하면 반드시 `super()`를 호출한다.** 특히 `on_fit_end`는 `super().on_fit_end()`가 threshold를 계산하므로(`adapters/base.py:132`), 모델 보정을 **먼저** 끝낸 뒤 호출한다.
- 모델별 lifecycle은 `TaskAdapter`(`src/core/adapter.py`)의 기존 hook으로 흡수한다. **공통 engine 시그니처를 바꾸지 않는다.** 추가 데이터 스트림이 필요하면 EfficientAD처럼 adapter가 직접 `build_dataloader`로 만들어 보관한다.
- 무작위성을 소비하는 검증 시점 계산(분위수 표본 추출 등)은 `torch.random.fork_rng`로 감싼다. 감싸지 않으면 valid split 크기가 학습 RNG를 교란해 재현성이 깨진다.
- 새 추상화를 `core/`로 올리지 않는다. 같은 필요가 **두 모델 이상에서 확인되기 전까지는** adapter 안에 둔다(NFR-005).

마지막으로 `src/tasks/anomaly/adapters/__init__.py`에 import와 `__all__` 항목을 추가한다. 등록 데코레이터는 import 시점에 실행되므로 이 줄이 빠지면 `RegistryError`가 난다.

### 3.7 단계 7 — config

v0.2의 config는 **data와 model이 직교**한다. 모델 추가 시 건드리는 것은 `configs/anomaly/models/<model>.yaml` 하나뿐이며, data config는 손대지 않는다.

```yaml
model:
  name: <model>_anomaly
  params:
    weights_path: ${paths.backbone_root}/<file>.pth

adapter:
  name: <model>
  params:
    smooth_sigma: 4.0

optim:
  optimizer:
    name: adam
    params: {lr: 0.0001}
  scheduler:
    name: step
    params: {step_size: 19, gamma: 0.1}   # paired with train.epochs below

selectors:
  <axis>:
    <value>:
      model.params.<key>: <resolved>
```

규칙:

- `_base`는 적지 않는다. data config가 `_base: ../_base.yaml`을 상속하며, model config는 그 위에 병합된다.
- **절대 경로를 직접 적지 않는다.** 자산 경로는 `${paths.dataset_root}` / `${paths.backbone_root}` placeholder를 쓴다. 머신별 값은 `configs/local.yaml`·환경변수·`--set`이 공급한다.
- `model.params`는 `MODELS.build`의 kwargs, `adapter.params`는 adapter 생성자 kwargs가 된다. adapter가 읽어야 하는 경로는 전부 `adapter.params`로 전달한다.
- 모델이 data 쪽 제약을 강제해야 하면(EfficientAD의 `batch_size: 1`, `normalize: false`) model config에 `data:` 블록으로 적고 **근거를 주석으로 남긴다.** 이 값이 data config를 덮는다.
- `train.epochs`처럼 모델 고유 기본값이 있으면 model config에 적는다.
- `selectors`는 CLI의 `--model.<axis> <value>`가 해석하는 치환 표다(`scripts/train.py:60`). backbone·size처럼 자주 바꾸는 축이 있으면 정의한다. 값 문자열에 `{value}` 치환도 쓸 수 있다(`configs/anomaly/data/mvtec.yaml` 참조).

### 3.8 단계 8 — 등록·스모크·무결성

체크리스트:

1. `src/tasks/anomaly/models/__init__.py`에 export 추가 (단계 5).
2. `src/tasks/anomaly/adapters/__init__.py`에 import 추가 (단계 6).
3. config 해석 확인 — 학습 전에 병합 결과를 먼저 본다.

   ```bash
   python scripts/train.py \
     --data configs/anomaly/data/mvtec.yaml --data.category bottle \
     --model configs/anomaly/models/<model>.yaml \
     --print_config
   ```

4. 소규모 스모크 — 단일 카테고리, 1~2 epoch로 train → evaluate → predict 3종을 통과시킨다.

   ```bash
   conda activate pytorch_env

   python scripts/train.py \
     --data configs/anomaly/data/mvtec.yaml --data.category bottle \
     --model configs/anomaly/models/<model>.yaml \
     --epochs 1 --run_name <model>_smoke

   python scripts/evaluate.py \
     --data configs/anomaly/data/mvtec.yaml --data.category bottle \
     --model configs/anomaly/models/<model>.yaml \
     --checkpoint outputs/<model>_smoke/checkpoints/best.pth \
     --run_name <model>_smoke_eval

   python scripts/predict.py \
     --data configs/anomaly/data/mvtec.yaml --data.category bottle \
     --model configs/anomaly/models/<model>.yaml \
     --checkpoint outputs/<model>_smoke/checkpoints/best.pth \
     --input <이미지 파일 또는 디렉터리> --run_name <model>_smoke_predict
   ```

   각 스크립트의 정확한 인자는 `docs/guides/cli-usage.md`를 확인한다.

5. 원본 무결성 — 기존 모델 파일에 변경이 없음을 확인한다.

   ```bash
   git status -- src/tasks/anomaly/models/
   ```

6. 공통 코드를 건드렸다면 **기존 모든 모델의 스모크를 재실행**한다. `core/`는 모든 task가 공유한다.

### 3.9 단계 9 — 성능 검증과 문서

**에이전트는 장시간 학습을 실행하지 않는다.** 실행 명령어를 제시하고 사용자 결과를 받는다.

- MVTec 대표 3개 카테고리(bottle, carpet, capsule)로 측정하고 anomalib reference와 비교한다. 여러 조건을 한 번에 돌리려면 `configs/anomaly/batch/`와 `scripts/batch.py`를 쓴다.
- 차이가 나면 모델 구현 차이로 단정하지 않고 순서대로 확인한다: preprocessing, optimizer, pretrained weight, normalization, threshold, metric, protocol.
- 기존 두 모델에서 실제 원인은 매번 **학습 예산**과 **calibration 시점**이었다. reference와 스텝 수가 자릿수로 다르면 그것을 먼저 의심한다.
- 원인이 무엇이든 원본 모델 코드를 고치지 않는다. adapter 또는 boilerplate에서 해결한다.

검증과 문서:

- 반대 벤더 CLI 적대적 검증을 받는다(`AGENTS.md`의 Cross-vendor Adversarial Review Sub-agent). Critical은 전부 수정하고 재검토한다. 실행은 대상당 최대 3회다.
- `docs/dev/v{major}.{minor}/reviews/A{n}.md`에 기록한다.
- 복사해 온 파일 목록(anomalib 원본 경로 → `src/tasks/anomaly/models/` 경로, diff 결과)을 진행 중인 버전의 `SPEC.md`에 기록한다.
- 문서 갱신 순서는 `SPEC.md → PLAN.md → backlog.json → PRD.md`다. 진행 중인 버전 폴더만 갱신한다.

## 4. 흔한 실패

| 증상 | 원인 | 조치 |
|---|---|---|
| 클론 `rev-parse HEAD`가 핀과 다르다 | 누군가 checkout·pull 했다 | 중단하고 사용자에게 보고(§1.1) |
| `ls`에 대상 모델 디렉터리가 없다 | sparse-checkout 미포함 | `sparse-checkout add` 승인 요청(§1.2) |
| `sparse-checkout add` 후에도 파일이 비어 있다 | partial clone의 blob fetch 실패(네트워크) | 네트워크 확인 후 재실행. 이 명령만은 오프라인 원칙의 예외이며 학습 자산이 아니다 |
| diff에서 import 외 차이가 대량 발생 | 클론 핀이 포팅 시점과 다르다 | §1.1 (1) 재확인 |
| `RegistryError: '<name>' is not registered` | `adapters/__init__.py` 또는 `models/__init__.py` import 누락 | 단계 8 체크리스트 1~2 |
| `RegistryError: ... is already registered` | 같은 이름을 두 번 register | `MODELS.register`는 alias용으로 `<model>`, `<model>_anomaly` 두 개만 |
| 실행 중 `OfflineViolationError` | 생성자가 pretrained 다운로드 시도 | no-download 팩토리(단계 5) |
| `LocalAssetError: ... unexpected keys` | 가중치 파일이 모델 구조와 불일치 | 올바른 `.pth` 경로 확인, 키 접두사 정리 로직 검토 |
| `${paths...}` 가 치환되지 않는다 | `configs/local.yaml` 미작성 | `configs/local.example.yaml` 복사 후 경로 기입 |
| epoch별 metric이 최종 평가와 어긋난다 | calibration이 validation 전에 일어나지 않음 | `on_validation_start`에서 보정 |
| 같은 seed인데 결과가 달라진다 | 검증 시점 계산이 전역 RNG를 소비 | `torch.random.fork_rng`로 감싼다 |
| optimizer가 frozen 파라미터를 잡는다 | freeze 시점이 `build_optimizer` 이후 | 팩토리에서 freeze |
| teacher가 매 epoch 학습 모드로 돌아간다 | engine의 `model.train()` | 인스턴스 수준 `train` 바인딩 고정 |

## 5. 범위 밖

**데이터셋 추가는 이 절차가 아니다.** 모델·adapter·`core/`를 건드리지 않고 아래 세 가지만 추가하면 되는 것이 정상이다.

1. `src/tasks/anomaly/datasets/<name>.py`에 `@DATASETS.register("<name>_anomaly")` 클래스 추가. 계약은 `MVTecAnomaly`와 같다 — train은 `(image, {})`, valid/test는 `(image, {"label", "mask"})`이며 정상 이미지도 all-zero mask를 반드시 포함한다.
2. `scripts/generate_splits.py`로 `configs/anomaly/splits/<name>_<category>.json` 생성.
3. `configs/anomaly/data/<name>.yaml` 추가.

이 범위를 넘어 모델 코드나 adapter를 고쳐야 한다면 데이터셋 추상화가 잘못된 것이다.
