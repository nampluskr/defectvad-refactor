# `defectvad`, `roi-corner-detection-ver3`, `cv_boilerplate`의 platform mechanism 비교

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

이 문서는 model이나 dataset 자체가 아닌 registry, factory/builder, config, checkpoint, runtime context, logging, error 및 offline 처리를 비교한다. 이 mechanism은 사용자가 새 anomaly model과 조건을 직접 추가할 수 있게 해야 하지만, model별 알고리즘 차이를 공통 core에 숨기는 수단이 되어서는 안 된다.

관련 model boundary는 [03_MODEL_AND_ADAPTER.md](03_MODEL_AND_ADAPTER.md), lifecycle/checkpoint 사용은 [04_EXECUTION_LIFECYCLE.md](04_EXECUTION_LIFECYCLE.md), CLI와 benchmark는 [06_CLI_AND_BATCH_ORCHESTRATION.md](06_CLI_AND_BATCH_ORCHESTRATION.md)에서 다룬다.

## 2. 분석 기준

분석 기준일은 2026-08-20이다. 정적 코드와 config만 확인했으며 local asset loading, offline block 또는 resume을 실행하지 않았다.

| 저장소 | Revision | platform 비교에서의 역할 |
|---|---|---|
| `defectvad` | `14879ea2a8970cee25438500e5abfeeb4be8e358` | module/class 문자열 factory와 model-state 중심 artifact의 레거시 근거 |
| `roi-corner-detection-ver3` | `8ae989a88996441e44fb2d5296a6419a8f661220` | 명시적 factory, config helper, output path와 run log의 사용성 근거 |
| `cv_boilerplate` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` | registry, validation, context, checkpoint, logger와 offline guard의 현재 근거 |

관련 요구사항은 `FR-003`, `FR-004`, `FR-008`, `FR-012`, `FR-021`~`FR-025`, `NFR-002`, `NFR-006`, `NFR-008`, `NFR-011`, `NFR-014`, `CON-006`, `CON-007`, `CON-011`이다.

## 3. 한눈에 보는 결론

| mechanism | `defectvad` | `roi-corner-detection-ver3` | `cv_boilerplate` |
|---|---|---|---|
| component lookup | config의 Python module/class를 `importlib`로 해석 | `get_wrapper` 조건문 | namespace별 registry logical name |
| config | 기본/dataset/model YAML 병합 | CLI defaults + Python dict | `_base` YAML inheritance + typed `--set` |
| early validation | import 또는 runtime 중심 | factory/argument path에서 오류 | required field, registry, path, device 사전 검사 |
| checkpoint | model `state_dict` | model file 및 history | model/optim/scheduler/scaler/RNG/config/env |
| runtime provenance | script logging 중심 | output dir 및 `run.log` | context env info, command line, metrics CSV |
| local/offline | 명시적 global guard 확인 못함 | 직접 지정 path 중심 | network guard와 local-weight error |

`권고` 공통 platform은 lookup, schema validation, reproducibility, artifact path 및 offline failure만 담당한다. upstream source, license, asset checksum과 model-specific lifecycle은 inventory/adapter/model package에 명시적으로 연결한다. Registry는 Python package 경로를 config에서 제거하지만, 등록만으로 upstream parity를 보장하지 않는다.

## 4. lookup과 config

`확인된 사실` `defectvad` factory는 config의 `module`을 import하고 `class`를 `getattr`로 찾아 dataset, model 및 trainer를 생성한다. config helper는 YAML을 읽고 여러 config dict를 병합한다.

근거:

- `defectvad@14879ea2:src/defectvad/common/factory.py#create_dataset`, `#create_model`, `#create_trainer`
- `defectvad@14879ea2:src/defectvad/common/config.py#load_config`, `#merge_configs`

`해석` 새 class의 등록 절차가 짧지만 package relocation, spelling 및 constructor contract 오류가 실행 시점까지 늦게 드러날 수 있다. config가 internal Python topology를 알아야 한다.

`확인된 사실` ROI factory `get_wrapper(model, ...)`는 model option에 대응하는 wrapper를 명시적으로 선택한다. common parser는 사용자가 model/network/head/dataset과 runtime size를 option으로 설정하게 하고 helper가 experiment name/output directory를 계산한다.

근거:

- `roi-corner-detection-ver3@8ae989a8:src/core/factory.py#get_wrapper`
- `roi-corner-detection-ver3@8ae989a8:scripts/config.py#parse_args`, `#get_exp_name`, `#get_output_dir`

`해석` 이 방식은 사용자에게 선택 축이 보이지만 새 model을 추가할 때 central condition을 반드시 수정해야 한다. model 이름 조건은 지속적으로 확장할 anomaly 공통 core에 적합하지 않다.

`확인된 사실` `cv_boilerplate`는 dataset, transform, model, loss, metric, adapter, builder namespace별 `Registry`를 제공한다. config validator는 top-level schema, required field/type, registry membership, data/weight path, monitor metric 및 CUDA availability를 검사한다. config는 제한된 `_base` inheritance와 existing-key-only dotted `--set` override를 지원한다.

근거:

- `cv_boilerplate@65d5412b:src/core/registry.py#Registry`
- `cv_boilerplate@65d5412b:src/core/config.py#resolve_config`, `#validate_config`, `#require_named`

`권고` anomaly model은 model registry entry와 adapter registry entry를 명시적으로 연결한다. 새 source model을 추가할 때 등록, config schema, local asset preflight, source manifest 및 reference protocol을 checklist로 만들며 registry name만 등록한 상태를 "가용"으로 판정하지 않는다.

## 5. checkpoint, context와 logging

`확인된 사실` `defectvad` `BaseModel.save/load`는 model state dict만 저장/복원한다. train script는 model 저장과 config 저장을 별도 option으로 처리한다.

근거: `defectvad@14879ea2:src/defectvad/common/base_model.py#BaseModel.save`, `#BaseModel.load`; `experiments/train.py#train`

`확인된 사실` ROI `Trainer.save`는 `history.json`을 저장하고 trainer/factory logger는 output directory에 `run.log`를 생성한다. 별도 command가 model file path를 checkpoint option으로 전달한다.

근거:

- `roi-corner-detection-ver3@8ae989a8:src/core/trainer.py#Trainer.save`
- `roi-corner-detection-ver3@8ae989a8:src/core/factory.py#get_logger`
- `roi-corner-detection-ver3@8ae989a8:scripts/config.py#get_checkpoint_path`

`확인된 사실` `cv_boilerplate` checkpoint는 model, optimizer, scheduler, AMP scaler, epoch, monitor/best metric, config, environment와 RNG state를 저장한다. `RunContext`는 seed, device, AMP, determinism, command line 및 environment metadata를 보유하고 metrics writer는 epoch/split/loss/metric/lr/elapsed row를 CSV에 append한다.

근거:

- `cv_boilerplate@65d5412b:src/core/checkpoint.py#save_checkpoint`, `#load_checkpoint`
- `cv_boilerplate@65d5412b:src/core/context.py#RunContext`
- `cv_boilerplate@65d5412b:src/core/logger.py#MetricsCsvWriter`

`해석` anomaly reference comparison에는 단순 score뿐 아니라 source revision, local asset identity, resolved protocol 및 checkpoint identity가 필요하다. current checkpoint/context는 이 정보를 담을 기반이지만 anomalib source/license/checksum과 map/threshold protocol의 field 완전성은 추가 audit이 필요하다.

## 6. local asset, offline 및 오류

`확인된 사실` `cv_boilerplate` entrypoint는 offline guard를 사용하며, guard는 loopback 외 socket connect/send/DNS resolution을 `OfflineViolationError`로 차단한다. `load_local_weights`는 path 부재와 strict state mismatch를 `LocalAssetError`로 보고하며 자동 download를 수행하지 않는다.

근거: `cv_boilerplate@65d5412b:src/core/offline.py#enable_offline_guard`, `#load_local_weights`

`확인된 사실` `defectvad`의 inspected common factory/config 및 ROI의 factory/config helper에는 이와 동등한 process-wide network guard를 확인하지 못했다. ROI는 explicit checkpoint path와 output directory를 CLI/config에서 받는다.

근거: `defectvad@14879ea2:src/defectvad/common/factory.py`; `roi-corner-detection-ver3@8ae989a8:scripts/config.py#parse_args`

`권고` anomaly integration은 local asset의 required/optional 여부, path, checksum, consumer와 source license를 preflight에서 검사한다. missing asset, unsupported source revision, registry name, output contract 또는 protocol은 actionable error로 fail하며 silent fallback, auto-download, default pretrained weight 선택을 하지 않는다.

## 7. 사용자 작업 지점과 미결정 사항

| 변경 목적 | 목표 변경 지점 | 확인할 영향 |
|---|---|---|
| 새 anomaly model 추가 | model/adapter registry, model config, inventory | source/license/asset/protocol/contract |
| 기존 model 조건 변경 | config + explicit `--set` | resolved config, output path, monitor/threshold |
| local weight 교체 | asset manifest와 config path | checksum, strict key match, offline run |
| resume/evaluate-only | checkpoint selector와 command | config identity, calibration buffer, split guard |
| batch 결과 비교 | benchmark manifest/runner | failure row, overwrite rule, provenance |

`미결정` registry 자동 import 범위, inventory의 storage format, local asset checksum algorithm, source diff approval workflow 및 long-running batch의 log retention은 현재 코드만으로 결정하지 않는다. 이 항목은 08 문서의 migration gate와 09 문서의 evidence index를 통해 후속 결정을 추적한다.

작성일: 2026-08-20  
상태: 세 저장소 정적 비교 초안
