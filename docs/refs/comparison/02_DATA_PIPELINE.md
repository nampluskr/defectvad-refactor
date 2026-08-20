# `defectvad`, `roi-corner-detection-ver3`, `cv_boilerplate`의 데이터 파이프라인 비교

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

이 문서는 image와 mask가 저장소에서 발견된 뒤 transform, split, collate와 `DataLoader`를 거쳐 model과 adapter에 전달되기까지의 경로를 비교한다.

비교 대상은 다음과 같다.

- dataset directory parsing과 sample identity
- train, valid와 test split의 표현 및 접근 경계
- image와 mask의 resize, dtype, normalization과 공간 정렬
- sample과 batch contract
- collate와 `DataLoader` 생성 정책
- dataset stage, CSV source, model, network, head와 실행 크기를 batch config에서 조립하는 방식
- EfficientAD의 auxiliary ImageNette stream처럼 primary dataset 밖에 있는 입력
- 새 dataset, transform 또는 실험 조건을 사용자가 직접 변경하는 위치

Metric 계산과 output post-processing은 [05_OUTPUT_AND_VISUALIZATION.md](05_OUTPUT_AND_VISUALIZATION.md)에서 다룬다. Model 내부의 auxiliary batch 소비와 lifecycle은 [03_MODEL_AND_ADAPTER.md](03_MODEL_AND_ADAPTER.md) 및 [04_EXECUTION_LIFECYCLE.md](04_EXECUTION_LIFECYCLE.md)에서 이어서 다룬다. `roi-corner-detection-ver3`의 subprocess 실행 순서, 실패 격리와 결과 수집은 [06_CLI_AND_BATCH_ORCHESTRATION.md](06_CLI_AND_BATCH_ORCHESTRATION.md)에서 상세히 비교하며, 이 문서는 dataset source와 실행 조건이 data pipeline에 연결되는 부분만 다룬다.

## 2. 분석 기준

분석 기준일은 2026-08-20이다.

| 구분 | 저장소 | Branch | Revision | 상태 |
|---|---|---|---|---|
| 사용자 bottom-up 구현 | `D:\_clones\defectvad` | `main` | `14879ea2a8970cee25438500e5abfeeb4be8e358` | clean |
| AI 에이전트 작성 boilerplate | `D:\_clones\cv_boilerplate` | `main` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` | clean |
| 사용자 운용 참고 구현 | `D:\_clones\roi-corner-detection-ver3` | `main` | `8ae989a88996441e44fb2d5296a6419a8f661220` | clean |

문장은 `확인된 사실`, `해석`, `권고`, `사용자 작업 지점`, `미결정`으로 구분한다. 세 저장소의 정적 코드, config와 canonical 문서만 조사했으며 dataset 로딩, 학습, 평가 및 benchmark는 실행하지 않았다. `roi-corner-detection-ver3`의 ROI corner geometry는 anomaly architecture의 근거가 아니라 사용자가 조건을 조립하고 반복 실행하는 방식의 근거로만 사용한다.

이 문서가 연결하는 주요 기준은 다음과 같다.

- 요구사항: `FR-007`, `FR-014`, `FR-015`, `FR-018`, `FR-022`, `FR-024`, `FR-025`, `NFR-009`, `CON-009`, `CON-010`
- 완료 기준: `AC-005`, `AC-006`, `AC-015`
- 설계: SPEC §4.2, §5, §13.2~§13.3
- 계획: `P0-T02`, `P1-T04`, `P3-T03`, `P4-T01`

## 3. 한눈에 보는 결론

```text
defectvad
YAML module/class
  -> factory가 image/mask transform 조립
  -> dataset이 directory를 직접 scan
  -> dict sample
  -> PyTorch 기본 collate
  -> config 그대로 DataLoader
  -> model별 trainer

cv_boilerplate
registry name + materialized split file
  -> train/eval transform을 registry에서 생성
  -> dataset이 split ID로 path와 target 해석
  -> (image, target)
  -> TaskAdapter의 collate
  -> 공통 builder가 seed와 split 접근 정책 적용
  -> 공통 Trainer와 adapter

roi-corner-detection-ver3
dataset stage + CSV 목록 + model/network/head
  -> 여러 CSV row를 하나의 sample list로 결합
  -> seed 기반 60:20:20 runtime split과 split별 size limit
  -> image/corner joint transform
  -> PyTorch 기본 collate와 공통 Dataloader 정책
  -> config별 train/evaluate/predict subprocess
```

| 질문 | `defectvad` | `cv_boilerplate` | `roi-corner-detection-ver3` |
|---|---|---|---|
| dataset 선택 | Python module과 class 경로 | `DATASETS` registry 이름 | 하나 이상의 `csv_path`; `dataset`은 논리 stage |
| 지원 split | `train`, `test` | `train`, `valid`, `test` | `train`, `valid`, `test`; predict는 split 없는 전체 목록 |
| split 표현 | directory 전체를 runtime scan | JSON의 materialized sample ID | 결합된 CSV row 순서와 seed로 runtime 60:20:20 재구성 |
| sample contract | anomaly metadata가 포함된 dict | `(image, target)` task contract | `(image, normalized corners)` 또는 image-only sample |
| joint geometry | image와 mask transform을 별도 조립 | torchvision v2가 image와 `Mask`를 함께 변환 | 같은 transform object가 image와 corner를 함께 변환 |
| collate | PyTorch 기본 collate | adapter가 `anomaly_collate` 제공 | PyTorch 기본 collate |
| loader 정책 | YAML 인자를 그대로 전달 | seed, worker seed, split guard와 device 정책을 공통 적용 | split별 shuffle/drop-last, generator, worker와 pin-memory 정책 |
| test leakage | train script가 test loader를 validation으로 전달할 수 있음 | train command는 test loader 생성을 차단하고 valid만 전달 | split은 분리하지만 materialized manifest와 test 접근 guard는 없음 |
| 조건 조립 | model trainer 내부 예외가 혼재 | config와 registry extension point | CSV, stage, model, network, head와 size를 config 한 건으로 조립 |

`해석` 목표 architecture의 데이터 기준은 `cv_boilerplate`의 materialized membership, task contract와 split 접근 정책이다. `roi-corner-detection-ver3`의 가치는 ROI corner target이나 runtime ratio split을 anomaly 공통 구조로 옮기는 데 있지 않다. 사용자가 CSV source, 논리 stage, model, network, head와 실행 크기를 읽을 수 있는 config에서 조립하고 같은 조건으로 lifecycle을 반복할 수 있다는 운용성을 anomaly inventory와 batch benchmark에 반영해야 한다.

## 4. `defectvad`의 bottom-up 데이터 파이프라인

### 4.1 `BaseDataset`이 anomaly sample 의미를 공통화한다

`확인된 사실` `BaseDataset`은 root, category, `train` 또는 `test` split, image transform과 mask transform을 받는다. 구체 dataset은 `_load_train_samples()`와 `_load_test_samples()`에서 다음 내부 sample record를 채운다.

```python
{
    "category": category,
    "image_path": image_path,
    "label": 0 또는 1,
    "defect_type": defect_type,
    "mask_path": mask_path 또는 None,
}
```

`__getitem__()`은 이를 다음 dict로 변환한다.

```python
{
    "image": image_tensor,
    "label": LongTensor,
    "defect_type": str,
    "mask": LongTensor[1, H, W],
    "dataset": str,
    "category": str,
    "filename": str,
}
```

근거: `defectvad@14879ea2:src/defectvad/data/base_dataset.py#BaseDataset`

`해석` MVTec, ViSA와 BTAD가 공통 anomaly sample dict를 사용하므로 레거시 안에서 dataset별 directory 차이는 격리된다. 반면 model, evaluator와 visualizer가 이 dict key와 mask shape를 직접 알기 때문에 contract가 anomaly 전용 base class 전반에 퍼진다.

### 4.2 MVTec directory parsing

`확인된 사실` `MVTecDataset`은 category별 다음 경로를 정렬해 읽는다.

```text
<root>/<category>/train/good/*.png
<root>/<category>/test/<defect_type>/*.png
<root>/<category>/ground_truth/<defect_type>/*_mask.png
```

Train sample은 모두 label 0과 `mask_path=None`이다. Test의 `good`은 label 0과 `mask_path=None`, 그 외 defect type은 label 1과 대응 mask path를 사용한다. 존재하지 않는 abnormal mask도 `None`으로 저장되어 이후 all-zero mask로 처리된다.

근거: `defectvad@14879ea2:src/defectvad/data/mvtec.py#MVTecDataset`

`확인된 사실` `BaseDataset._load_mask()`는 mask path가 없으면 image transform 결과의 높이와 너비를 사용해 zero mask를 만든다. mask가 있으면 grayscale image를 transform한 뒤 `mask > 0`으로 이진화한다.

근거: `defectvad@14879ea2:src/defectvad/data/base_dataset.py#BaseDataset._load_mask`

`해석` 정상 evaluation image에 명시적인 zero mask를 제공하는 의미는 `FR-014`와 일치한다. 그러나 abnormal mask 파일 누락도 정상 mask와 같은 zero tensor로 조용히 바뀌므로 잘못된 dataset을 즉시 실패시키지 못한다.

### 4.3 Factory가 transform까지 조립한다

`확인된 사실` `create_dataset()`은 YAML의 `module`과 `class`를 import한 뒤 image와 mask transform을 직접 만든다.

```python
image = Resize -> optional CenterCrop -> ToTensor -> optional Normalize
mask  = Resize -> optional CenterCrop -> ToTensor
```

Image normalization의 on/off, mean과 std는 dataset config에 있다. Image와 mask는 별도 `Compose`지만 같은 `img_size`와 `crop_size`를 사용한다.

근거:

- `defectvad@14879ea2:src/defectvad/common/factory.py#create_dataset`
- `defectvad@14879ea2:configs/defaults.yaml#dataset`
- `defectvad@14879ea2:configs/datasets/mvtec.yaml#dataset`

`해석` 한 config에서 geometry를 맞추기 쉬운 구조지만 dataset 생성과 preprocessing 선택이 factory에 결합되어 있다. `torchvision.transforms.Resize`의 기본 interpolation을 mask에 별도로 지정하지 않고 resize 후 `> 0`을 적용하므로, binary mask 가장자리의 보간값이 foreground로 확장될 수 있다. 이는 SPEC §5.4의 nearest mask interpolation보다 약한 보장이다.

### 4.4 명시적인 validation split이 없다

`확인된 사실` dataset class와 factory가 허용하는 split은 `train`과 `test`뿐이다. `experiments/train.py`는 `test_dataset`과 `test_loader`를 만든 뒤 trainer validation이 활성화되면 그 loader를 `valid_loader`로 전달한다.

```python
train_dataset = create_dataset("train", config["dataset"])
test_dataset = create_dataset("test", config["dataset"])
...
trainer.fit(train_loader, valid_loader=test_loader)
```

근거:

- `defectvad@14879ea2:src/defectvad/data/base_dataset.py#BaseDataset.__init__`
- `defectvad@14879ea2:experiments/train.py#train`

`해석` reference 구현과 같은 test-as-validation protocol을 재현할 수는 있지만 해당 선택이 manifest나 protocol flag로 드러나지 않는다. Validation 결과가 model state, quantile 또는 threshold에 영향을 주는 모델에서는 final test data 누수 여부를 실행 기록만으로 구분하기 어렵다.

### 4.5 `DataLoader`는 config를 그대로 전달한다

`확인된 사실` `create_dataloader()`는 `DataLoader(dataset, **config)`만 호출한다. Train과 test의 batch size, shuffle, drop-last, worker와 pin-memory 설정은 각각 YAML에 있다. 별도 collate가 없으므로 dict sample은 PyTorch 기본 collate에 의해 batched dict가 된다.

근거:

- `defectvad@14879ea2:src/defectvad/common/factory.py#create_dataloader`
- `defectvad@14879ea2:configs/defaults.yaml#train_loader`
- `defectvad@14879ea2:configs/defaults.yaml#test_loader`

`해석` 사용자가 loader 인자를 직접 이해하고 바꾸기 쉽다. 그러나 worker seed, device에 따른 pin-memory, persistent worker, train/test 접근 제한 같은 실행 정책은 공통 factory가 보장하지 않는다.

### 4.6 여러 dataset과 model별 예외

`확인된 사실` ViSA와 BTAD도 `BaseDataset`의 sample dict를 재사용한다. ViSA는 CSV의 split과 label을 읽고, BTAD는 `ok`, `ko` directory 및 category별 mask 확장자를 해석한다.

근거:

- `defectvad@14879ea2:src/defectvad/data/visa.py#ViSADataset`
- `defectvad@14879ea2:src/defectvad/data/btad.py#BTADDataset`

`확인된 사실` EfficientAD는 공통 primary loader 외에 trainer 내부에서 `DATASET_DIR/imagenette2`를 찾고 `ImageFolder`와 별도 transform 및 `DataLoader(batch_size=1)`를 만든다. `training_step()`은 iterator를 순환시키며 primary image와 ImageNette image를 model에 함께 전달한다. Train loader의 batch size가 1인지 검사하지만 normalization 금지 검사는 주석 처리되어 있다.

근거: `defectvad@14879ea2:src/defectvad/models/efficientad/model_trainer.py#EfficientADTrainer`

`해석` 레거시는 EfficientAD penalty stream이라는 실제 요구를 구현했지만 경로, transform, loader와 iterator lifecycle이 model trainer 안에 숨는다. 사용자가 config만 보고 required asset과 두 입력의 관계를 파악하기 어렵다.

## 5. `cv_boilerplate`의 데이터 파이프라인

### 5.1 Registry와 config가 생성 경계를 나눈다

`확인된 사실` `MVTecAnomaly`는 `@DATASETS.register("mvtec_anomaly")`로 등록된다. `build_transforms()`는 train과 eval transform spec을 `TRANSFORMS` registry에서 각각 만들고, `build_dataset()`은 `DATASETS` registry에 root, split, transform과 data params를 전달한다. File split이면 config의 path를 `split_path`로 넘긴다.

근거:

- `cv_boilerplate@65d5412b:src/tasks/anomaly/dataset.py#MVTecAnomaly`
- `cv_boilerplate@65d5412b:src/cli/commands.py#build_transforms`
- `cv_boilerplate@65d5412b:src/cli/commands.py#build_dataset`
- `cv_boilerplate@65d5412b:src/tasks/anomaly/__init__.py`

```yaml
data:
  name: mvtec_anomaly
  root: /mnt/d/datasets/mvtec
  params: {category: bottle}
  image_size: [256, 256]
  batch_size: 8
  split:
    mode: file
    path: configs/splits/mvtec_bottle.json
  transform:
    train: {name: anomaly_default, params: {}}
    eval:  {name: anomaly_default, params: {}}
```

근거: `cv_boilerplate@65d5412b:configs/anomaly/_base.yaml`

`해석` dataset parser, transform 선택과 loader 정책을 독립적으로 교체할 수 있다. 대신 새 module의 decorator가 실행되도록 task package import까지 연결해야 한다.

### 5.2 Materialized sample ID가 split membership을 결정한다

`확인된 사실` `MVTecAnomaly`는 directory 전체를 split으로 사용하지 않는다. JSON에 기록된 ID를 정렬해 해당 split의 sample 목록으로 사용한다.

```text
train ID: train_good/<stem>
valid/test ID: <defect_type>/<stem>
```

`train_good` prefix는 MVTec train/good과 test/good의 같은 stem 충돌을 피한다. 현재 `mvtec_bottle.json`은 train 209개, valid 33개, test 50개 ID를 가진다.

근거:

- `cv_boilerplate@65d5412b:src/tasks/anomaly/dataset.py#TRAIN_PREFIX`
- `cv_boilerplate@65d5412b:src/tasks/anomaly/dataset.py#MVTecAnomaly`
- `cv_boilerplate@65d5412b:configs/splits/mvtec_bottle.json`

`확인된 사실` `load_split_file()`과 dataset constructor는 train, valid와 test 집합의 교집합이 비었는지 검사한다. CLI train command는 train과 valid만 만들며, 공통 loader builder는 명시적 `allow_test_split=True` 없이 test loader 생성을 거부한다. Evaluate command는 해당 opt-in을 사용한다.

근거:

- `cv_boilerplate@65d5412b:src/data/split.py#assert_disjoint`
- `cv_boilerplate@65d5412b:src/core/builders.py#build_dataloader`
- `cv_boilerplate@65d5412b:src/cli/commands.py#train`
- `cv_boilerplate@65d5412b:src/cli/commands.py#evaluate`

`해석` 레거시의 암묵적인 test-as-validation 경로가 명시적인 valid와 test 접근 경계로 바뀌었다. 현재 materialized split은 leakage-safe한 실행을 지원하지만 anomalib reference와 동일한 split인지에 대한 protocol 결정은 별도 문제다.

### 5.3 Sample contract와 mask 의미

`확인된 사실` train은 `(image, {})`를 반환한다. Valid와 test는 다음 target을 반환한다.

```python
{
    "label": LongTensor scalar,  # 0=normal, 1=anomalous
    "mask": LongTensor[H, W],   # values in {0, 1}
}
```

Abnormal mask가 없거나 읽히지 않으면 예외가 발생한다. Normal evaluation image는 원본 image 크기의 all-zero mask를 받는다.

근거: `cv_boilerplate@65d5412b:src/tasks/anomaly/dataset.py#MVTecAnomaly.__getitem__`

`해석` mask 누락을 정상 sample처럼 조용히 처리하지 않고 실패시키며, normal image도 pixel metric 모집단에 포함할 수 있다. Mask channel은 레거시의 `[1,H,W]`에서 `[H,W]`로 바뀌었고 adapter가 batch에서 `[B,H,W]`로 stack한다.

### 5.4 Joint transform이 geometry와 interpolation을 보장한다

`확인된 사실` `AnomalyTransform`은 image와 target mask를 한 번에 받는다. Mask를 `tv_tensors.Mask`로 감싼 뒤 같은 torchvision v2 `Compose`에 전달하므로 resize geometry는 공유되고 mask에는 nearest interpolation이 선택된다. Normalize는 image에만 적용된다.

```python
Resize(image_size, antialias=True)
ToDtype(Image=float32, Mask=int64, scale=True)
Normalize(IMAGENET_MEAN, IMAGENET_STD)
```

Train의 빈 target과 predict의 target 없는 image도 같은 image transform 경로를 사용한다.

근거: `cv_boilerplate@65d5412b:src/tasks/anomaly/transform.py#AnomalyTransform`

`해석` image와 mask에 같은 숫자 설정을 두 번 기술하던 레거시보다 공간 정렬과 mask interpolation이 contract로 드러난다.

### 5.5 Adapter가 collate를 선택한다

`확인된 사실` `anomaly_collate()`는 image를 `[B,3,H,W]` tensor로 stack하고 target은 list-of-dict로 유지한다. `AnomalyAdapter.collate_fn()`이 이 함수를 공통 loader builder에 제공한다.

근거:

- `cv_boilerplate@65d5412b:src/tasks/anomaly/collate.py#anomaly_collate`
- `cv_boilerplate@65d5412b:src/tasks/anomaly/adapter.py#AnomalyAdapter.collate_fn`

`해석` 공통 engine은 anomaly target의 내부 key를 알지 않고, train의 빈 dict와 evaluation의 label/mask dict를 같은 batch 외형으로 전달한다. 새 dataset의 target이 이 contract와 같다면 collate와 engine을 바꿀 필요가 없다.

### 5.6 공통 builder가 loader 정책을 적용한다

`확인된 사실` `build_dataloader()`는 다음을 공통 적용한다.

- train만 shuffle
- config의 batch size, worker와 train drop-last
- adapter의 collate function
- worker init seed와 별도 `torch.Generator` seed
- CUDA일 때만 pin-memory
- worker가 있을 때 persistent workers
- test split 생성에 명시적 opt-in 요구

근거: `cv_boilerplate@65d5412b:src/core/builders.py#build_dataloader`

`해석` loader 생성이 단순 parameter 전달에서 재현성과 split 사용 정책을 포함하는 platform mechanism으로 확장됐다.

### 5.7 CLI와 notebook의 조립 경로

`확인된 사실` CLI train은 다음 순서로 data pipeline을 조립한다.

```text
resolve/validate config
  -> build train/eval transforms
  -> build train/valid datasets
  -> bind class names
  -> build train/valid loaders
  -> Trainer.fit(loaders={train, valid})
```

근거: `cv_boilerplate@65d5412b:src/cli/commands.py#train`

`확인된 사실` `notebooks/tasks/04_anomaly.ipynb`는 교육을 위해 `DATASETS.build()`, `TRANSFORMS.build()`와 PyTorch `DataLoader`를 직접 호출한다. 이 notebook의 loader는 adapter collate는 사용하지만 공통 builder의 worker seed, generator, test guard와 device별 pin-memory 정책은 거치지 않는다.

근거: `cv_boilerplate@65d5412b:notebooks/tasks/04_anomaly.ipynb`

`권고` 구조 탐색 cell에서는 직접 조립을 유지할 수 있지만 재현 실험 cell은 `src.cli.commands`와 `src.core.builders`의 조립 순서를 사용하고 resolved config를 함께 보존한다.

## 6. `roi-corner-detection-ver3`의 사용자 운용 데이터 파이프라인

### 6.1 CSV가 sample source를 명시한다

`확인된 사실` `CornerDataset`은 하나 이상의 UTF-8 CSV에서 `image_dir`, `image_name`과 `x1,y1,...,x4,y4`를 읽는다. CSV 목록 순서와 각 파일의 row 순서대로 하나의 sample list를 만들며, corner는 `TL`, `TR`, `BR`, `BL` 순서의 normalized `[4,2]` 좌표다.

근거:

- `roi-corner-detection-ver3@8ae989a8:src/data/dataset.py#Dataset._load_csv`
- `roi-corner-detection-ver3@8ae989a8:src/data/dataset.py#CornerDataset._parse_row`
- `roi-corner-detection-ver3@8ae989a8:docs/guides/01-dataset-format.md#6.-여러-CSV-결합`

`확인된 사실` `dataset` CLI 값은 실제 sample source를 자동 선택하지 않는다. 실제 source는 `csv_path`와 CSV 각 행의 `image_dir`, `image_name`이 결정하며, `dataset`은 output 경로와 이전 stage checkpoint 연결에 쓰이는 논리 stage다.

근거:

- `roi-corner-detection-ver3@8ae989a8:scripts/config.py#DEFAULTS`
- `roi-corner-detection-ver3@8ae989a8:scripts/config.py#get_output_dir`
- `roi-corner-detection-ver3@8ae989a8:scripts/config.py#get_prev_checkpoint_path`

`해석` 사용자가 여러 source를 명시적인 목록으로 조립할 수 있다는 점은 anomaly dataset inventory와 benchmark suite에 참고할 가치가 있다. Corner column과 polygon 순서는 ROI task 전용 contract이므로 anomaly sample contract로 이전하지 않는다.

### 6.2 Runtime split은 재구성 가능하지만 materialized provenance는 아니다

`확인된 사실` factory는 결합된 전체 sample list를 같은 seed로 두 번 나눠 train 60%, valid 20%, test 20%를 만든다. 그 뒤 `train_size`, `valid_size`, `test_size`가 지정되면 각 split에서 다시 seed 기반 subset을 선택한다.

근거:

- `roi-corner-detection-ver3@8ae989a8:src/data/dataset.py#BaseDataset.split`
- `roi-corner-detection-ver3@8ae989a8:src/data/dataset.py#BaseDataset.subset`
- `roi-corner-detection-ver3@8ae989a8:src/core/factory.py#get_dataset`
- `roi-corner-detection-ver3@8ae989a8:src/core/factory.py#get_dataloader`

`확인된 사실` 여러 CSV를 먼저 결합한 뒤 전체 목록을 분할하므로 source별 stratification이나 group split은 제공하지 않는다. CSV 목록 또는 row 순서가 바뀌면 같은 seed에서도 membership이 달라진다.

`해석` 이 방식은 사용자가 seed와 size를 쉽게 바꾸는 운용성은 보여주지만 `CON-009`, `CON-010`과 reference protocol의 증거에는 부족하다. 목표 anomaly workflow는 `cv_boilerplate`의 materialized sample ID, disjoint 검사와 test 접근 guard를 유지해야 한다.

### 6.3 Joint transform은 target-aware preprocessing의 참고 사례다

`확인된 사실` train transform은 resize, horizontal flip, vertical flip, rotation, color jitter, Gaussian blur, tensor conversion과 ImageNet normalization을 순서대로 적용한다. Geometry transform은 image와 corner를 함께 변경하고, valid와 test는 resize, tensor conversion과 normalization만 적용한다.

근거:

- `roi-corner-detection-ver3@8ae989a8:src/core/factory.py#get_transform`
- `roi-corner-detection-ver3@8ae989a8:src/data/transforms.py#Compose`
- `roi-corner-detection-ver3@8ae989a8:src/data/transforms.py#RandomHorizontalFlip`
- `roi-corner-detection-ver3@8ae989a8:src/data/transforms.py#RandomVerticalFlip`
- `roi-corner-detection-ver3@8ae989a8:src/data/transforms.py#RandomRotation`

`해석` image와 target geometry를 함께 변경한다는 원칙은 anomaly mask 정렬과 같은 방향이다. 다만 normalized corner를 위한 좌표 재배열, 범위 검사와 회전 계산은 ROI 전용이며, anomaly mask의 nearest interpolation과 binary semantics를 대체하지 않는다.

### 6.4 Dataloader 정책과 batch condition 조립

`확인된 사실` `Dataloader`는 train에서만 shuffle과 drop-last를 사용한다. 별도 `torch.Generator`에 seed를 설정하고 CUDA 가용 시 pin-memory를 사용하며, worker가 있으면 persistent worker와 prefetch factor 4를 설정한다. 별도 collate function은 제공하지 않는다.

근거: `roi-corner-detection-ver3@8ae989a8:src/data/dataloader.py#Dataloader.__init__`

`확인된 사실` `configs/public.py`와 `configs/synthetic.py`는 stage별 CSV 목록, batch size, epoch와 split size를 `BASE`로 정의한 뒤 `configs/models.py`의 model, network, head 조합과 병합해 `CONFIGS`를 만든다. Batch runner는 각 config를 train, evaluate, predict CLI argument로 변환하고 subprocess별 성공과 실패를 기록한 뒤 다음 config를 계속 실행한다.

근거:

- `roi-corner-detection-ver3@8ae989a8:configs/models.py#MODELS`
- `roi-corner-detection-ver3@8ae989a8:configs/public.py#BASE`
- `roi-corner-detection-ver3@8ae989a8:configs/public.py#CONFIGS`
- `roi-corner-detection-ver3@8ae989a8:configs/synthetic.py#CONFIGS`
- `roi-corner-detection-ver3@8ae989a8:scripts/batch_run.py#PASS_KEYS`
- `roi-corner-detection-ver3@8ae989a8:scripts/batch_run.py#get_cli_args`
- `roi-corner-detection-ver3@8ae989a8:scripts/batch_run.py#run`

`권고` SOTA anomaly inventory의 실행 단위도 dataset source와 manifest, model identifier, preprocessing protocol, local asset, lifecycle 조건과 실행 크기를 사용자가 읽고 수정할 수 있는 config로 표현한다. ROI 프로젝트의 `model`, `network`, `head` 명칭을 그대로 강제하지 말고 anomaly model capability와 adapter config에 대응시킨다.

## 7. 구성요소별 직접 비교

| 구성요소 | `defectvad` | `cv_boilerplate` | `roi-corner-detection-ver3` | 목표 판단 |
|---|---|---|---|---|
| parser | dataset subclass가 directory/CSV 해석 | task dataset이 directory와 split ID 해석 | 여러 CSV row를 순서대로 결합 | parser는 dataset adapter에 두고 membership과 provenance를 분리 |
| sample identity | filename과 dataset metadata | split ID를 내부에 보유 | CSV row 위치와 image path에 암묵적으로 의존 | stable ID와 source path를 batch에 전달 |
| split | train/test directory 의미 | materialized train/valid/test ID | seed 기반 runtime 60:20:20 | materialized manifest와 test guard 유지 |
| sample 외형 | batched dict | `(image, target)` | `(image, corners)` | anomaly task contract 유지 |
| joint target transform | image와 mask를 별도 transform | image와 `Mask`를 v2로 함께 변환 | image와 corner를 custom joint transform으로 변환 | target-aware 원칙만 재사용 |
| collate | PyTorch 기본 collate | adapter collate | PyTorch 기본 collate | adapter가 task batch 의미를 소유 |
| loader | YAML kwargs 전달 | 공통 policy builder | 공통 `Dataloader`와 CLI/config 인자 | boilerplate builder에 정책을 집중하고 사용자 override 노출 |
| validation | test loader 재사용 가능 | valid loader 분리 | runtime valid/test 분리 | materialized protocol과 test 접근 제한 필요 |
| auxiliary data | EfficientAD trainer가 직접 생성 | 현재 경로 없음 | 복수 primary CSV 결합만 지원 | named auxiliary loader contract 필요 |
| batch 조건 | trainer별 YAML | 단일 run config 중심 | `BASE`와 model/network/head 조합을 `CONFIGS`로 전개 | anomaly inventory와 benchmark matrix로 표현 |

## 8. 현재 구현의 충족 사항과 gap

### 8.1 확인된 충족 사항

| 항목 | 현재 근거 | 관련 기준 |
|---|---|---|
| MVTec normal/anomaly와 mask parsing | `MVTecAnomaly.__getitem__` | `FR-014`, `AC-006` 일부 |
| 정상 evaluation image의 zero mask | `MVTecAnomaly.__getitem__` | `FR-014`, `AC-006` 일부 |
| binary mask와 nearest resize | dataset과 `AnomalyTransform` | `FR-014`, SPEC §5.4 |
| materialized train/valid/test | `mvtec_bottle.json` | `CON-010` 일부 |
| split disjoint 검사 | `assert_disjoint` | `CON-009`, `AC-015` 일부 |
| train command의 test loader 차단 | `build_dataloader`와 `train` | `CON-009`, `AC-015` 일부 |
| dataset/transform 등록 경계 | `DATASETS`, `TRANSFORMS` | `FR-015`, `NFR-009` 일부 |
| 재현 가능한 loader seed | `build_dataloader`, `make_worker_init_fn` | `NFR-002` 일부 |

정적 구조가 기준을 지원한다는 뜻이며 acceptance가 완료됐다는 뜻은 아니다. Dataset fixture, split provenance와 실제 image/mask 정렬 결과는 실행 검증되지 않았다.

### 8.2 `DATA-GAP-001`: manifest provenance와 전체 모집단 검증 부족

`확인된 사실` 현재 split JSON은 train, valid와 test ID 배열만 가진다. SPEC §5.3이 요구하는 dataset identity, category, seed, source population과 validation/test 공유 flag를 기록하지 않는다.

`확인된 사실` anomaly dataset은 split 간 disjoint만 검사한다. Classification dataset이 호출하는 `assert_subset()`을 MVTec에는 적용하지 않으므로 unknown ID는 sample load 시점에 발견되고, directory에 있지만 manifest에서 누락된 sample은 검출되지 않는다.

`영향` `FR-022`, `CON-010`, `AC-006`의 완전한 증거로 사용하기 부족하다.

### 8.3 `DATA-GAP-002`: ratio split schema와 construction 불일치

`확인된 사실` config validation은 `data.split.mode`의 `file`과 `ratio`를 허용하고 `generate_ratio_split()`도 존재한다. 그러나 CLI `build_dataset()`은 file mode에만 `split_path`를 전달하며 `MVTecAnomaly`는 항상 split file을 요구한다. `generate_ratio_split()`의 호출 지점은 현재 source에 없다.

`영향` anomaly config에서 ratio mode는 validation을 통과할 수 있지만 dataset construction에서 실패한다. 현재 지원 protocol은 사실상 file mode뿐이다.

### 8.4 `DATA-GAP-003`: dataset-independent identity가 batch에 전달되지 않음

`확인된 사실` `MVTecAnomaly`는 `self.ids`에 stable ID를 보관하지만 valid/test target에는 label과 mask만 넣는다. SPEC §4.2의 `sample_id`, `path`, optional metadata는 batch에 전달되지 않는다.

`영향` metric sample, visualization, error와 source file의 역추적이 dataset object 밖에서는 어렵다. `FR-013`, `FR-015`, `NFR-008`의 관찰 가능성과 연결된다.

### 8.5 `DATA-GAP-004`: 모델별 preprocessing이 아직 표현되지 않음

`확인된 사실` STFPM과 EfficientAD config는 같은 anomaly base config를 상속한다. 두 모델 모두 `anomaly_default`의 ImageNet normalization과 batch size 8을 사용하며 EfficientAD config는 model weight path만 override한다.

`확인된 사실` 레거시 EfficientAD trainer는 train batch size 1을 요구하고 ImageNette penalty stream에 normalization 없는 transform을 사용한다. Primary normalization 금지 검사는 주석 처리되어 있어 레거시 자체도 완전한 reference 보장은 아니다.

`영향` `FR-007`, `AC-005`와 이미 PRD `GAP-004`에 기록된 차이가 현재 checkout에도 남아 있다.

### 8.6 `DATA-GAP-005`: auxiliary dataset construction 경로 부재

`확인된 사실` 현재 `cv_boilerplate` source와 config에는 `auxiliary`, `imagenette` 또는 `penalty` data specification이 없다. EfficientAD model도 auxiliary batch를 받지 않아 레거시의 ImageNette penalty loss 경로가 보존되지 않는다.

`영향` SPEC §5.5의 named auxiliary loader mapping은 목표 설계이며 현재 구현이 아니다. `FR-008`, `P3-T03`, `P4-T01`에서 해결해야 한다.

### 8.7 `DATA-GAP-006`: 자동화된 data contract 증거 부재

`확인된 사실` 현재 `cv_boilerplate@65d5412b` checkout에는 `tests/` directory가 없다. Notebook과 source 주석은 존재하지만 `AC-006`의 parsing, zero mask, binary mask, geometry와 split 검증을 자동으로 증명하지 않는다.

`영향` 현재 문서는 구조를 확인했을 뿐 acceptance 완료를 선언할 수 없다.

### 8.8 `DATA-GAP-007`: notebook과 CLI loader 정책 차이

`확인된 사실` anomaly task notebook은 PyTorch `DataLoader`를 직접 만든다. CLI는 공통 builder를 사용한다.

`영향` 같은 dataset과 transform을 사용해도 seed, worker, pin-memory와 split guard가 달라질 수 있다. 교육용 직접 조립과 재현 실험 경로를 구분해야 한다.

### 8.9 `DATA-GAP-008`: anomaly inventory와 data protocol의 batch matrix 부재

`확인된 사실` `roi-corner-detection-ver3`는 stage별 CSV 목록과 실행 크기를 model, network, head 조합에 병합해 사용자가 수정 가능한 `CONFIGS`를 만든다. 현재 `cv_boilerplate`에는 가용하다고 판정된 모든 SOTA anomaly model inventory와 dataset, split manifest, preprocessing, auxiliary asset 및 reference protocol을 조합한 동등한 batch matrix가 없다.

근거:

- `roi-corner-detection-ver3@8ae989a8:configs/models.py#MODELS`
- `roi-corner-detection-ver3@8ae989a8:configs/public.py#CONFIGS`
- `roi-corner-detection-ver3@8ae989a8:scripts/batch_run.py#run`

`영향` 새 SOTA 모델을 추가할 때 model code뿐 아니라 dataset source, preprocessing과 reference protocol을 어떤 조합으로 검증해야 하는지 사용자가 한 위치에서 확인하기 어렵다. `FR-018`, `FR-022`, `FR-024`, `FR-025`와 연결된다.

`권고` 고정된 model list가 아니라 model inventory와 benchmark suite를 분리하고, 각 실행 case가 model ID, dataset inventory ID, materialized split, preprocessing protocol, local asset와 reference protocol ID를 참조하게 한다.

### 8.10 `DATA-GAP-009`: 논리 stage와 실제 dataset identity의 구분 미확정

`확인된 사실` ROI 프로젝트의 `dataset` 값은 실제 sample source가 아니라 output 및 이전 checkpoint 연결에 쓰이는 논리 stage이고 실제 source는 `csv_path`가 정한다.

근거:

- `roi-corner-detection-ver3@8ae989a8:scripts/config.py#get_output_dir`
- `roi-corner-detection-ver3@8ae989a8:scripts/config.py#get_prev_checkpoint_path`
- `roi-corner-detection-ver3@8ae989a8:docs/guides/01-dataset-format.md#7.-Default-CSV와-명시적-CSV`

`영향` anomaly workflow에서 dataset identity, training stage와 benchmark suite를 하나의 `dataset` 문자열에 합치면 split provenance와 staged lifecycle 의미가 혼동될 수 있다.

`미결정` target config에서 dataset inventory ID, benchmark suite ID와 optional lifecycle stage를 별도 축으로 둘지는 상위 설계에서 결정해야 한다.

## 9. 이전 판정

| 레거시 요소 | 판정 | 목표 위치 | 근거와 조건 |
|---|---|---|---|
| MVTec directory와 mask naming 해석 | 조정 | `src/tasks/anomaly/dataset.py` | 핵심 의미는 유지하되 manifest, fail-fast와 tuple contract 사용 |
| 정상 image의 zero mask | 재사용 | anomaly dataset target | pixel metric 모집단 보존에 필요 |
| `BaseDataset` inheritance | 제외 | 없음 | registry와 `(image, target)` contract로 dataset별 parser를 수용 가능 |
| anomaly dict 전체 | 대체 | dataset target와 `TaskAdapter` | engine과 metric의 dataset key 결합을 줄임 |
| factory 내부 image/mask Compose | 대체 | transform registry와 joint v2 transform | geometry, interpolation과 model config 경계 개선 |
| YAML `train_loader`/`test_loader` kwargs 전달 | 대체 | 공통 `build_dataloader` | seed와 split policy를 함께 적용 |
| test를 validation으로 전달하는 경로 | 제외 | explicit valid/test protocol | 승인된 exact-reference run이면 별도 protocol로 명시해야 함 |
| ViSA와 BTAD parser | 미결정 | 향후 anomaly dataset module | 초기 MVTec 범위 밖이며 실제 사용자 범위 결정 필요 |
| EfficientAD ImageNette transform과 iterator | 조정 | named auxiliary data spec과 adapter/model lifecycle | penalty 의미는 보존하되 trainer 내부 path와 loader 생성을 이전하지 않음 |
| 복수 CSV source 목록 | 조정 | dataset inventory 또는 benchmark suite config | 명시적 source 조립 경험은 유지하되 dataset identity와 provenance를 함께 기록 |
| seed 기반 runtime 60:20:20 split | 제외 | materialized split manifest | reference protocol과 row-order 독립성을 보장하지 못함 |
| image/corner joint transform | 조정 | anomaly target-aware transform | joint geometry 원칙만 재사용하고 corner 계산은 이전하지 않음 |
| `BASE`와 model/network/head의 `CONFIGS` 전개 | 조정 | anomaly inventory와 benchmark matrix | 사용자 조건 조립 경험을 anomaly capability와 protocol ID로 표현 |
| 논리 dataset stage와 이전 checkpoint 연결 | 미결정 | lifecycle 또는 benchmark config | dataset identity와 분리할 축이 필요한지 결정 |

`권고` auxiliary loader가 하나의 model에만 필요하더라도 dataset, transform과 loader 생성은 기존 registry와 builder를 재사용한다. 소비 시점과 iterator cadence만 adapter 또는 model-specific lifecycle에 둔다. ROI 프로젝트의 `model`, `network`, `head`는 이름 자체가 아니라 독립적인 조건 축을 조립하는 사용성을 참고하며 공통 engine에 model명 분기를 추가하지 않는다.

## 10. 사용자가 직접 변경하는 위치

### 10.1 MVTec category 또는 split 변경

`사용자 작업 지점`

1. `configs/anomaly/<model>.yaml` 또는 CLI `--set`에서 `data.params.category`를 바꾼다.
2. 해당 category용 materialized split JSON을 준비한다.
3. `data.split.path`를 그 파일로 바꾼다.
4. split ID가 실제 directory와 mask에 대응하고 세 집합이 disjoint인지 검증한다.
5. resolved config와 split manifest를 실행 결과에 함께 보존한다.

Category만 바꾸고 bottle split 파일을 그대로 사용하면 stem과 defect type이 우연히 존재하더라도 잘못된 protocol이 될 수 있다. 현재 manifest에는 category metadata가 없으므로 사용자가 두 값을 함께 관리해야 한다.

### 10.2 Transform 또는 image size 변경

등록된 transform의 parameter만 바꾸려면 config의 `data.image_size`와 `data.transform.train/eval.params`를 사용한다. 새 transform 의미가 필요하면 다음을 변경한다.

1. `src/tasks/anomaly/transform.py`에 builder를 구현한다.
2. `@TRANSFORMS.register("<name>")`로 등록한다.
3. `src/tasks/anomaly/__init__.py`의 import 연결을 유지한다.
4. image와 mask geometry, nearest mask interpolation과 eval/predict 동일성을 검증한다.
5. model config에서 transform name을 선택한다.

Normalization을 dataset class에 넣지 않는다. EfficientAD처럼 model별 입력 제약이 있으면 model config가 별도 transform을 선택해야 한다.

### 10.3 새 anomaly dataset 추가

1. `src/tasks/anomaly/dataset.py` 또는 별도 module에 `Dataset`을 구현한다.
2. `@DATASETS.register("<name>")`로 등록한다.
3. anomaly package에서 module을 import한다.
4. Train은 `(image, {})`, evaluation은 공통 label/mask target을 반환한다.
5. stable sample ID와 source path 전달 방식을 정의한다.
6. explicit split membership과 asset 누락의 fail-fast 동작을 구현한다.
7. 기존 `anomaly_collate`와 `AnomalyAdapter`를 그대로 쓸 수 있는지 먼저 확인한다.
8. dataset fixture로 normal/abnormal mask, geometry와 split을 검증한다.

### 10.4 Loader 조건 변경

Batch size, worker와 drop-last는 config에서 변경한다.

```text
python -m src train configs/anomaly/stfpm.yaml \
  --set data.batch_size=4 \
  --set data.num_workers=2 \
  --set data.drop_last=false
```

이 명령은 parser와 config override 구조를 근거로 작성했으며 실행하지 않았다. Shuffle, seed와 test opt-in은 공통 policy이므로 일반 실험 조건 때문에 `build_dataloader()`를 수정하지 않는다.

`해석` ROI 프로젝트는 batch config에서 `batch_size`, `num_workers`, `train_size`, `valid_size`, `test_size`를 case별로 전달한다. 목표 anomaly workflow도 사용자가 실행 크기를 case별로 바꿀 수 있어야 하지만 split size 변경이 승인된 materialized membership을 암묵적으로 다시 생성하게 해서는 안 된다.

### 10.5 Auxiliary dataset 추가

현재는 config key나 builder 경로가 없으므로 단순 설정 변경만으로 추가할 수 없다. 구현 시에는 SPEC §5.5에 따라 다음 연결이 필요하다.

```text
model config의 named auxiliary spec
  -> local asset preflight
  -> TRANSFORMS/DATASETS registry
  -> 공통 DataLoader builder
  -> loaders["<name>"]
  -> adapter가 선언된 이름만 model lifecycle에 전달
```

경로를 model trainer의 환경 변수로 숨기거나 공통 engine에서 `efficientad` 이름으로 분기하지 않는다.

### 10.6 SOTA anomaly model과 dataset protocol을 batch case로 추가

현재 checkout에는 전체 SOTA inventory를 data protocol과 조합하는 batch config가 없으므로 아래 내용은 목표 `사용자 작업 지점`이다.

1. Model inventory에 upstream source, revision, license, local asset와 lifecycle capability를 등록한다.
2. Dataset inventory에 root, category 또는 subset, target semantics와 stable sample identity를 등록한다.
3. 승인된 materialized split manifest와 provenance를 연결한다.
4. Model별 primary transform과 named auxiliary data spec을 연결한다.
5. Benchmark suite에서 model ID, dataset ID, split protocol ID와 reference protocol ID를 하나의 case로 조립한다.
6. CLI override는 batch size, worker와 제한된 실행 크기처럼 protocol을 바꾸지 않는 값에 사용한다.
7. Resolved case와 실제 asset identity를 결과 artifact에 보존한다.

`권고` ROI 프로젝트처럼 사용자가 한 config에서 조건 조합을 읽고 수정할 수 있어야 한다. 그러나 CSV row 순서에 의존하는 runtime split, ROI corner head 명칭과 논리 stage 문자열을 anomaly contract로 그대로 복제하지 않는다.

## 11. CLI와 notebook 사용 경계

`권고` 재현 가능한 train/evaluate는 CLI와 resolved config를 기준으로 한다.

```text
python -m src config configs/anomaly/stfpm.yaml
python -m src train configs/anomaly/stfpm.yaml
python -m src evaluate configs/anomaly/stfpm.yaml --checkpoint <local-checkpoint> --split valid
```

명령의 정확한 argument와 artifact 경로는 [06_CLI_AND_BATCH_ORCHESTRATION.md](06_CLI_AND_BATCH_ORCHESTRATION.md)에서 검토한다. 위 명령은 실행하지 않았다.

Notebook에서 data sample을 탐색할 때는 registry를 직접 사용할 수 있다. 재현 실험에서는 다음 순서를 유지한다.

```text
offline guard
  -> src.tasks import
  -> resolve_config와 validate_config
  -> build_transforms
  -> build_dataset
  -> adapter 생성
  -> 공통 build_dataloader
```

## 12. 미결정 사항과 권고 순서

| ID | 미결정 또는 후속 조치 | 영향 | 연결 |
|---|---|---|---|
| `DATA-D01` | disjoint final protocol과 exact anomalib reference protocol을 병행할지 승인 | split과 benchmark 의미 | `P0-T02`, `GAP-009` |
| `DATA-D02` | split manifest metadata schema와 source population 검증 방식 | provenance와 누락 검출 | SPEC §5.3, `P1-T04` |
| `DATA-D03` | `sample_id`, path와 metadata의 target 내 위치 | 추적성과 collate contract | SPEC §4.2, `P1-T01` |
| `DATA-D04` | ratio mode를 구현할지 현재 schema에서 제거할지 | config truthfulness | SPEC §13.3 |
| `DATA-D05` | EfficientAD의 정확한 input normalization, batch와 auxiliary protocol | reference fidelity | `FR-007`, `FR-008`, `P4-T01` |
| `DATA-D06` | ViSA와 BTAD를 초기 migration 범위에 포함할지 | dataset coverage | `FR-015` |
| `DATA-D07` | dataset inventory ID, benchmark suite ID와 lifecycle stage를 별도 축으로 둘지 결정 | provenance와 staged execution 의미 | `FR-018`, `FR-022`, SPEC §11.2, §13.4 |
| `DATA-D08` | SOTA model inventory와 data protocol을 어떤 batch matrix schema로 조립할지 결정 | 사용자가 model과 조건을 추가하는 경로 | `FR-024`, `FR-025`, `P5-T01`, `P5-T02` |
| `DATA-D09` | source별 stratification 또는 group split이 필요한 dataset 범위를 결정 | leakage와 dataset 독립성 | `FR-015`, `CON-009`, `CON-010` |

권고 순서는 다음과 같다.

1. `P0-T02`에서 split과 model별 input protocol을 승인한다.
2. `P1-T04`에서 manifest metadata, population membership와 MVTec fixture를 확정한다.
3. `P3-T03`에서 named auxiliary loader contract를 만든다.
4. `P4-T01`에서 EfficientAD transform, batch size와 ImageNette penalty stream을 reference와 대조한다.
5. 실제 결과로 `AC-005`, `AC-006`, `AC-015`를 검증한다.
6. `P5-T01`과 `P5-T02`에서 model inventory, dataset inventory와 reference protocol을 사용자 수정 가능한 benchmark matrix로 연결한다.

## 13. 요구사항 추적

| 요구사항 또는 gap | 이 문서의 관련 절 | 현재 판단 |
|---|---|---|
| `FR-007` 모델별 preprocessing | §4.3, §5.4, §8.5, §10.2 | 구조적 extension은 있으나 EfficientAD protocol 미충족 |
| `FR-008` auxiliary stream | §4.6, §8.6, §10.5 | 레거시 근거 확인, 목표 구현 부재 |
| `FR-014` MVTec | §4.2, §5.2~§5.4 | 핵심 parsing 의미 구현, 실행 검증 필요 |
| `FR-015`, `NFR-009` dataset 독립성 | §5.1, §5.5, §6.1~§6.2, §10.3 | registry 경계 존재, identity 전달 보완 필요 |
| `FR-018` benchmark orchestration | §6.4, §8.9, §10.6 | ROI 조건 조립 경험은 확인, 목표 anomaly batch matrix는 미구현 |
| `FR-022` 재현 정보와 provenance | §6.1~§6.2, §8.2, §8.9~§8.10 | materialized split의 source와 protocol metadata 보완 필요 |
| `FR-024` 새 모델 추가 | §6.4, §8.9, §10.6 | registry 경계는 있으나 지속 가능한 SOTA inventory와 data matrix 부재 |
| `FR-025` 실패 격리 | §6.4, §8.9 | ROI batch runner는 참고 근거이며 상세 판정은 문서 06에서 수행 |
| `CON-009` test 누수 금지 | §4.4, §5.2, §6.2 | ROI runtime split은 대안이 아니며 boilerplate test guard 유지 필요 |
| `CON-010` 명시적 split | §5.2, §6.2, §8.2 | ID membership은 명시, metadata와 population 검증 부족 |
| `AC-005` model-specific protocol | §8.5~§8.6 | 미충족 |
| `AC-006` dataset contract | §8.1~§8.2, §8.7 | 정적 구조 일부 충족, fixture 증거 없음 |
| `AC-015` leakage 방지 | §5.2, §8.1 | 구조적 guard 존재, 승인 protocol 검증 필요 |
| `GAP-004` preprocessing | §8.5 | 현재 checkout에서도 확인 |
| `GAP-005` auxiliary lifecycle | §8.6 | 현재 checkout에서도 확인 |
| `GAP-009` split protocol | §5.2, §12 | 미결정 유지 |
| `DATA-GAP-008` batch matrix | §6.4, §8.9, §10.6 | 사용자 수정 가능한 SOTA model/data/protocol matrix 미구현 |
| `DATA-GAP-009` stage와 identity | §6.1, §8.10, §12 | dataset identity와 lifecycle stage 분리 방식 미결정 |

작성일: 2026-08-20  
상태: 세 저장소 비교 초안
