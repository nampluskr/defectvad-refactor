# MODEL-ADD — anomalib 모델 추가 절차

상위 문서: [BRIEF.md](../BRIEF.md) · [PRD.md](../PRD.md) · [SPEC.md](../SPEC.md) · [PLAN.md](../PLAN.md)

P6-T01 산출물이다. v0.1에서 STFPM·EfficientAD 두 모델을 통합하며 확립된 구조를 기준으로, 세 번째 모델을 추가하는 절차를 이 문서만 보고 수행할 수 있도록 정리한다. [BRIEF.md](../BRIEF.md) "공통 절차" 6단계를 실제 경험으로 보정한 결과이며, 충돌 시 이 문서가 우선한다.

## 1. 전제

- 대상 anomalib은 `anomalib@091ca6a`(v2.3.0, 로컬 `/mnt/d/projects/clones/anomalib`)이다. 다른 commit을 쓰려면 [UPSTREAM-INVENTORY.md](UPSTREAM-INVENTORY.md)를 먼저 갱신한다.
- 대상은 `src/anomalib/models/image/<model>/`의 image 모델로 한정한다(BRIEF).
- 세 원칙은 예외 없다. upstream 무수정(CON-001), Lightning 미사용(CON-002), 자동 다운로드 금지(CON-003·CON-004).
- 실행 전 conda 환경 `pytorch_env`를 활성화한다.

## 2. 절차 요약

| 단계 | 작업 | 산출물 |
|---|---|---|
| 1 | 대상 모델 파일 구성 조사 | 복사 대상 목록 |
| 2 | 의존 모듈 추적 (Lightning 오염 검사 포함) | 추가 복사 대상 |
| 3 | `upstream/<model>/`에 복사 + import 경로만 치환 | `upstream/` 신규 파일 |
| 4 | `lightning_model.py` 정독 → 이관표 작성 | config 항목 / adapter hook 목록 |
| 5 | `adapters/<model>.py` 작성 (adapter + 모델 팩토리) | adapter, `MODELS` 엔트리 |
| 6 | `configs/anomaly/<model>.yaml` 작성 | config |
| 7 | 등록·스모크·무결성 확인 | 동작 확인 |
| 8 | 성능 검증 (사용자 실행) | reference 비교 |
| 9 | 적대적 검증 + 문서 갱신 | `reviews/A{n}.md`, 인벤토리 갱신 |

## 3. 단계별 상세

### 3.1 단계 1 — 파일 구성 조사

`ls /mnt/d/projects/clones/anomalib/src/anomalib/models/image/<model>/`로 구성을 확인한다. 모델마다 다르다.

- STFPM: `torch_model.py`, `loss.py`, `anomaly_map.py`, `lightning_model.py`
- EfficientAD: `torch_model.py`, `lightning_model.py` (loss·map이 `torch_model.py` 안에 있다)

복사 대상은 **`lightning_model.py`와 `__init__.py`를 제외한 순수 PyTorch 파일 전부**다. `__init__.py`는 복사하지 않는다 — anomalib 패키지 `__init__`은 Lightning 경로를 끌어온다(§3.2).

### 3.2 단계 2 — 의존 모듈 추적

복사 대상 파일의 모든 `from anomalib...` import를 추적한다. v0.1에서 실제로 문제가 된 두 함정이 그대로 재현될 가능성이 높다(SPEC §3).

1. **`components/__init__.py`를 복사하면 Lightning이 딸려 온다.** `components/base/anomalib_module.py`가 `import lightning.pytorch as pl`을 한다. 따라서 패키지 `__init__`이 아니라 **실제로 필요한 모듈 파일만** 옮기고, import는 모듈 단위로 직접 건다.
2. **`from anomalib.data import InferenceBatch`는 `anomalib.data.__init__`을 통해 Lightning `DataModule`에 닿는다.** 이미 `upstream/components/data/torch_base.py`로 복사되어 있으므로 새 모델은 그 경로를 재사용한다.

검사 방법: 복사한 파일 트리에서 `grep -rn "lightning" src/tasks/anomaly/upstream/`을 실행해 docstring 외 실행 경로 언급이 없는지 확인한다(현재 남은 4건은 모두 docstring이며, 지우면 CON-001 위반이다).

이미 `upstream/components/`에 있는 재사용 가능한 모듈은 [UPSTREAM-INVENTORY.md](UPSTREAM-INVENTORY.md) §2를 참조한다. 새 components가 필요하면 같은 규칙으로 파일째 복사한다. **필요한 클래스만 발췌하지 않는다** — 파일 단위 복사가 CON-001의 diff 검증을 가능하게 한다.

### 3.3 단계 3 — 복사와 import 치환

```text
src/tasks/anomaly/upstream/<model>/     # anomalib 원본 — 수정 금지 구역
```

허용되는 변경은 import 경로 치환 **한 가지뿐**이다.

```python
from anomalib.data import InferenceBatch
# ->
from src.tasks.anomaly.upstream.components.data.torch_base import InferenceBatch
```

복사 직후 원본과 diff해 import 외 차이가 0라인인지 확인하고 기록한다(AC-001과 동일한 검증).

```bash
diff -u /mnt/d/projects/clones/anomalib/src/anomalib/models/image/<model>/torch_model.py \
        src/tasks/anomaly/upstream/<model>/torch_model.py
```

### 3.4 단계 4 — `lightning_model.py` 이관표

`lightning_model.py`는 **복사하지 않고 읽는다.** 읽으면서 아래 표를 채운다. 이 표가 단계 5·6의 작업 지시서가 된다.

| anomalib `lightning_model.py` | 이 프로젝트에서의 자리 |
|---|---|
| `configure_optimizers` | `configs/anomaly/<model>.yaml`의 `optim` (코드 아님) |
| `training_step`의 loss 조합 | `<Model>Adapter.train_step` |
| `on_train_start` | `<Model>Adapter.on_fit_start` |
| `on_validation_start` | `<Model>Adapter.on_validation_start` |
| `on_train_end` / 학습 후 보정 | `<Model>Adapter.on_fit_end` (`super()` 호출 필수) |
| `prepare_*` 계열의 다운로드 | **다운로드 분기는 옮기지 않는다.** 로컬 경로 strict load로 대체 |
| `trainer_arguments`의 제약(batch size 등) | config에 값으로 기재 + 주석으로 근거 |
| post-processing / normalization | 공통 `AnomalyAdapter` + `postprocess.py`로 이미 충족되는지 먼저 확인 |

optimizer 이관에서 주의할 점(SPEC §4.2): `build_optimizer`는 `requires_grad=True`인 파라미터로 **단일 parameter group**만 만든다. anomalib이 특정 서브모듈만 optimizer에 넣는다면, 그 집합이 freeze 결과와 일치하는지 확인한다. 일치하면 config만으로 충분하다. 서로 다른 lr을 갖는 복수 group이 필요하면 `core/builders.py`에 group 지정 방식을 추가해야 하며, 이는 공통 코드 변경(등급 C)이므로 사용자 승인이 필요하다.

scheduler의 `step_size`처럼 `train.epochs`에서 파생되는 값은 v0.1에 참조 문법이 없으므로 config에 상수로 적고 **짝이 되는 키를 주석으로 명시**한다(`configs/anomaly/efficientad.yaml` 참조).

### 3.5 단계 5 — `adapters/<model>.py`

한 파일에 두 가지를 둔다. 두 모델 모두 이 형태다.

```python
@ADAPTERS.register("<model>")
class <Model>Adapter(AnomalyAdapter):
    def train_step(self, model, batch, device): ...
    # 필요한 lifecycle hook만 override

@MODELS.register("<model>_anomaly")
def build_<model>(weights_path, **params):
    # no-download 생성 + 로컬 가중치 주입 -> upstream nn.Module 반환
```

작성 규칙:

- `AnomalyAdapter`를 상속하고 **모델별 차이만 override**한다. metric·threshold·smooth·visualize는 공통 구현을 그대로 쓴다.
- `train_step`은 `{"loss": Tensor, "loss_dict": dict}`를 반환한다. loss 계산은 upstream 모듈을 호출한다(`STFPMLoss` 등). loss를 새로 작성하지 않는다.
- `eval_step`/`predict_step`은 보통 override하지 않는다. upstream이 `InferenceBatch`를 반환해도 `postprocess.py#to_output_dict`가 `(B,)`/`(B,H,W)`로 변환한다(SPEC §4.2).
- **hook을 override하면 반드시 `super()`를 호출한다.** 특히 `on_fit_end`는 `super().on_fit_end()`가 threshold를 계산하므로, 모델 보정을 **먼저** 끝낸 뒤 호출한다(SPEC §4.3).
- 모델별 lifecycle은 `TaskAdapter`의 기존 hook으로 흡수한다. **공통 engine 시그니처를 바꾸지 않는다.** 추가 데이터 스트림이 필요하면 EfficientAD처럼 adapter가 직접 `build_dataloader`로 만들어 보관한다(SPEC §4.5).
- 무작위성을 소비하는 검증 시점 계산(분위수 표본 추출 등)은 `torch.random.fork_rng`로 감싼다. 감싸지 않으면 valid split 크기가 학습 RNG를 교란해 재현성이 깨진다(AC-009).
- 새 추상화를 `core/`로 올리지 않는다. 같은 필요가 **두 모델 이상에서 확인되기 전까지는** adapter 안에 둔다(NFR-005).

no-download 팩토리(SPEC §4.6)는 upstream 생성자가 pretrained를 건드리는지에 따라 두 갈래다.

| 상황 | 처리 | 예 |
|---|---|---|
| 생성자가 pretrained를 로드/다운로드한다 | 진입점 함수(`timm.create_model` 등)를 `pretrained=False`로 덮는 wrapper로 **생성자 호출 한 문장 동안만** 치환하고 `try/finally`로 원복 → 이후 로컬 `.pth` 주입 | `build_stfpm` |
| 생성자가 가중치를 건드리지 않는다 | 팩토리는 경로만 모델 속성에 보관하고, `on_fit_start`에서 `load_local_weights(..., strict=True)` | `build_efficientad` |

이 치환은 **`adapters/` 밖으로 내보내지 않는다.** `core/`에 일반화된 "pretrained 차단 컨텍스트"를 만들지 않는다.

strict load 판정(SPEC §4.6):

- `missing`이 하나라도 있으면 `LocalAssetError`로 즉시 실패한다.
- `unexpected`는 **모델이 실제로 갖고 있지 않은 최상위 서브모듈 이름의 키만** 허용한다(timm이 `out_indices` 밖 스테이지를 제거하므로 `fc.`/`head.` 접두사 허용 목록으로는 부족하다). 모델이 갖고 있는 이름인데도 unexpected면 구조 불일치이므로 실패시킨다.
- 어떤 경우에도 조용한 랜덤 초기화 폴백을 두지 않는다(CON-004).

frozen 파라미터 점검:

- teacher 등 학습 대상이 아닌 서브모듈은 `requires_grad=False`가 **`build_optimizer` 호출 전**에 설정되어야 optimizer param group에서 빠진다. 팩토리 시점에 설정한다.
- `nn.ParameterDict.update()`는 새 `nn.Parameter`(기본 `requires_grad=True`)로 교체하므로, 통계·분위수 버퍼는 **update 할 때마다 다시 freeze**한다.
- upstream이 `train()`을 override해 eval을 고정하지 않는 모델은, 공통 engine의 `model.train()`이 매 epoch teacher를 학습 모드로 되돌린다. 인스턴스 수준 `train` 바인딩으로 고정한다(`build_efficientad` 참조). upstream 파일은 건드리지 않는다.

마지막으로 `src/tasks/anomaly/adapters/__init__.py`에 새 모듈 import를 추가한다. 등록 데코레이터는 import 시점에 실행되므로 이 한 줄이 빠지면 `RegistryError`가 난다.

### 3.6 단계 6 — config

`configs/anomaly/<model>.yaml`은 `_base.yaml`을 상속하고 차이만 적는다.

```yaml
_base: _base.yaml
model:
  name: <model>_anomaly
  params:
    weights_path: ${paths.backbone_root}/<file>.pth
adapter:
  name: <model>
  params: {}
optim:
  optimizer: {name: ..., params: {...}}
  scheduler: ...
```

규칙:

- **절대 경로를 직접 적지 않는다.** 자산 경로는 `${paths.dataset_root}` / `${paths.backbone_root}` placeholder를 쓴다(SPEC §6). 머신별 값은 `configs/local.yaml`·환경변수·`--set`이 공급한다.
- `adapter.params`의 값은 그대로 adapter 생성자 kwargs가 된다(`cli/commands.py:91`). adapter가 읽어야 하는 경로는 전부 여기로 전달한다.
- `model.params`는 `MODELS.build`의 kwargs가 된다(`cli/commands.py:85`).
- 새 자산이 필요하면 `configs/assets.yaml`의 `datasets`/`weights`에 항목을 추가한다. 그래야 `check-assets`가 새 머신에서 누락을 잡아낸다.
- optimizer 빌더가 registry에 없으면 `core/builders.py`에 범용 빌더로 추가한다(모델명 분기가 아니면 NFR-005 위반이 아니다).

### 3.7 단계 7 — 등록·스모크·무결성

체크리스트:

1. `src/tasks/anomaly/adapters/__init__.py`에 import 추가 (단계 5).
2. 소규모 스모크 — 단일 카테고리, 1~2 epoch로 train → evaluate → predict 3종을 통과시킨다.

```bash
conda activate pytorch_env
python -m src check-assets
python -m src train configs/anomaly/<model>.yaml --set train.epochs=1 --set output.run_name=<model>_smoke
python -m src evaluate configs/anomaly/<model>.yaml \
    --checkpoint outputs/runs/anomaly/<model>_smoke/checkpoints/best.pth --split test \
    --set output.run_name=<model>_smoke_eval
python -m src predict configs/anomaly/<model>.yaml \
    --checkpoint outputs/runs/anomaly/<model>_smoke/checkpoints/best.pth \
    --input <이미지 파일 또는 디렉터리> --set output.run_name=<model>_smoke_predict
```

3. upstream 무결성 — `git status -- src/tasks/anomaly/upstream/`로 기존 파일에 변경이 없음을 확인한다.
4. 공통 코드 오염 검사 — `python scripts/check_engine_purity.py`.
5. 공통 코드를 건드렸다면 **기존 모든 모델의 스모크를 재실행**한다. `core/`는 5개 task가 공유한다.

### 3.8 단계 8 — 성능 검증

에이전트는 장시간 학습을 실행하지 않는다. 실행 명령어를 제시하고 사용자 결과를 받는다.

- MVTec 대표 3개 카테고리(bottle, carpet, capsule)로 측정하고 anomalib reference와 비교한다.
- 차이가 나면 모델 구현 차이로 단정하지 않고 PRD §5.4 항목을 순서대로 확인한다: preprocessing, optimizer, pretrained weight, normalization, threshold, metric, protocol.
- v0.1 경험상 실제 원인은 **학습 예산**과 **calibration 시점**이었다. reference와 스텝 수가 자릿수로 다르면 그것을 먼저 의심한다.
- 원인이 무엇이든 upstream을 고치지 않는다. adapter 또는 boilerplate에서 해결한다.

### 3.9 단계 9 — 검증과 문서

- 반대 벤더 CLI 적대적 검증을 받는다(Claude 구현 → Codex 검토, Codex 구현 → Claude Sonnet 검토). Critical은 전부 수정하고 재검토한다. 검증 실행은 대상당 최대 3회다.
- `docs/dev/v{major}.{minor}/reviews/A{n}.md`에 기록한다.
- [UPSTREAM-INVENTORY.md](UPSTREAM-INVENTORY.md)에 새 파일 행을 추가한다.
- 문서 갱신 순서는 `SPEC.md → PLAN.md → backlog.json → PRD.md`다.

## 4. 데이터셋 추가 (v0.2 대비)

모델 추가와 축이 다르다. 모델·adapter·`core/`를 건드리지 않고 아래 세 가지만 추가하면 되는 것이 정상이다.

1. `src/tasks/anomaly/dataset.py`에 `@DATASETS.register("<name>_anomaly")` 클래스를 추가한다. 계약은 `MVTecAnomaly`와 같다 — train은 `(image, {})`, valid/test는 `(image, {"label", "mask"})`, 정상 이미지도 all-zero mask를 반드시 포함한다.
2. split 생성 스크립트를 `scripts/generate_<name>_splits.py`로 추가하고 `configs/splits/<name>_<category>.json`을 만든다. `scripts/check_split_integrity.py`로 disjoint를 확인한다.
3. config에서 `data.name`·`data.root`·`data.split.path`만 바꾼다. `configs/assets.yaml`에도 항목을 추가한다.

이 범위를 넘어 모델 코드나 adapter를 고쳐야 한다면 데이터셋 추상화가 잘못된 것이다(PLAN §4.1).

## 5. 흔한 실패

| 증상 | 원인 | 조치 |
|---|---|---|
| `RegistryError: '<name>' is not registered` | `adapters/__init__.py` import 누락 | 단계 5 마지막 항목 |
| 실행 중 `OfflineViolationError` | 생성자가 pretrained 다운로드 시도 | no-download 팩토리(단계 5) |
| `LocalAssetError: ... missing required keys` | 가중치 파일이 모델 구조와 불일치 | 올바른 `.pth` 경로 확인, `key_map` 검토 |
| `ConfigError: paths...` | `configs/local.yaml` 미작성 또는 root 부재 | `configs/local.example.yaml` 복사 후 경로 기입 |
| epoch별 metric이 최종 평가와 어긋난다 | calibration이 validation 전에 일어나지 않음 | `on_validation_start`에서 보정(SPEC §4.3) |
| 같은 seed인데 결과가 달라진다 | 검증 시점 계산이 전역 RNG를 소비 | `torch.random.fork_rng`로 감싼다 |
| optimizer가 frozen 파라미터를 잡는다 | freeze 시점이 `build_optimizer` 이후 | 팩토리에서 freeze |

---

작성일: 2026-08-21
문서 상태: P6-T01 산출물
