# 비교 문서 evidence index

## 1. 목적과 사용법

이 문서는 [01~08 비교 문서](README.md)의 주요 판단을 코드 근거로 역추적하기 위한 색인이다. 각 항목은 분석 기준일 2026-08-20의 정적 확인 결과다. 성능, numerical parity, license 적합성 또는 실행 성공을 뜻하지 않는다.

근거 표기 형식은 `저장소@revision:상대경로#symbol`이다. symbol이 없는 경우 해당 파일과 확인한 구성요소를 적었다. 실제 수정 또는 구현 전에는 현재 checkout과 revision이 같은지 확인하고, 다르면 해당 항목을 재검증해야 한다.

| 별칭 | 저장소 | Revision |
|---|---|---|
| DV | `D:\_clones\defectvad` | `14879ea2a8970cee25438500e5abfeeb4be8e358` |
| ROI | `D:\_clones\roi-corner-detection-ver3` | `8ae989a88996441e44fb2d5296a6419a8f661220` |
| CVB | `D:\_clones\cv_boilerplate` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` |

## 2. 전체 구조와 사용자 운용

| 판단 | 분류 | 근거 | 연결 문서 |
|---|---|---|---|
| DV 실행 script가 config, model/trainer, dataset/loader, fit와 evaluator를 조립한다. | 확인된 사실 | `DV:experiments/train.py#train` | 01 §4.1, 04 §4 |
| DV factory가 config의 Python module/class를 import/getattr로 해석한다. | 확인된 사실 | `DV:src/defectvad/common/factory.py#create_model`, `#create_trainer` | 01 §4.2, 07 §4 |
| ROI는 model/network/head/dataset option과 common CLI를 제공한다. | 확인된 사실 | `ROI:scripts/config.py#DEFAULTS`, `#parse_args`, `#get_wrapper_kwargs` | 01 §5.1, 06 §5 |
| ROI batch runner는 case failure를 수집한 뒤 다음 case를 계속 실행한다. | 확인된 사실 | `ROI:scripts/batch_run.py#run` | 01 §5.1, 06 §5 |
| CVB는 command dispatch 전에 task registration을 import하고 train/evaluate/predict/benchmark command를 분기한다. | 확인된 사실 | `CVB:src/__main__.py#dispatch`, `src/cli/commands.py#train` | 01 §6.1 |
| ROI task-specific target/head은 anomaly architecture의 source of truth가 아니다. | 권고 | 사용자 의도, comparison README §7.3 | 01, 02, 08 |

## 3. data와 batch contract

| 판단 | 분류 | 근거 | 연결 문서 |
|---|---|---|---|
| DV `BaseDataset`은 image, label, mask와 anomaly metadata를 dict sample로 제공한다. | 확인된 사실 | `DV:src/defectvad/data/base_dataset.py#BaseDataset` | 02 §4.1 |
| DV MVTec loader는 train good, test defect type와 ground-truth mask path를 해석한다. | 확인된 사실 | `DV:src/defectvad/data/mvtec.py#MVTecDataset` | 02 §4.2 |
| DV train script가 test loader를 validation loader로 전달할 수 있다. | 확인된 사실 | `DV:experiments/train.py#train` | 02 §4.4, 04 §4 |
| ROI factory는 CSV source와 runtime split/size로 dataset/loader를 만든다. | 확인된 사실 | `ROI:src/core/factory.py#get_dataset`, `#get_dataloader`, `#get_predict_dataloader` | 02 §6 |
| CVB data loader는 adapter collate와 split 접근 정책을 받는다. | 확인된 사실 | `CVB:src/core/builders.py#build_dataloader` | 02 §7 |
| target/mask geometry의 최종 anomaly contract는 source audit 전 미결정이다. | 미결정 | SPEC §5와 model별 upstream source 필요 | 02, 03, 05 |

## 4. model과 adapter

| 판단 | 분류 | 근거 | 연결 문서 |
|---|---|---|---|
| DV `BaseModel`은 Tensor/dict/DataLoader prediction과 model-state save/load를 제공한다. | 확인된 사실 | `DV:src/defectvad/common/base_model.py#BaseModel` | 01 §4.3, 03 §4 |
| DV `BaseTrainer`는 model-specific `training_step`과 internal evaluator validation을 연결한다. | 확인된 사실 | `DV:src/defectvad/common/base_trainer.py#BaseTrainer` | 03 §4, 04 §4 |
| ROI `BaseWrapper`는 model, pre/postprocessor, optimizer, loss와 metric을 묶고 raw output을 final corner로 변환한다. | 확인된 사실 | `ROI:src/models/base/wrapper.py#BaseWrapper` | 03 §5 |
| ROI factory의 model-name 조건문은 new model extension의 central modification point다. | 확인된 사실 | `ROI:src/core/factory.py#get_wrapper` | 03 §5, 07 §4 |
| CVB `TaskAdapter`는 train/eval/predict, metrics, batch size와 optional hook을 contract로 정의한다. | 확인된 사실 | `CVB:src/core/adapter.py#TaskAdapter` | 03 §6 |
| common engine에는 model/task name branch를 두지 않고 adapter가 anomaly semantics를 해석해야 한다. | 권고 | AGENTS.md 프로젝트 방향; `CVB:src/core/engine.py#Trainer` | 03 §6~§7, 08 §2 |

## 5. lifecycle과 state

| 판단 | 분류 | 근거 | 연결 문서 |
|---|---|---|---|
| DV evaluate/predict는 config와 model/dataset을 새로 만들고 weight를 load한다. | 확인된 사실 | `DV:experiments/evaluate.py#evaluate`, `experiments/predict.py#predict` | 04 §4 |
| ROI trainer는 wrapper hook과 train/eval step을 호출하고 optional early-stop best state를 메모리에서 복원한다. | 확인된 사실 | `ROI:src/core/trainer.py#Trainer` | 04 §5 |
| CVB fit은 valid monitor로 best/last checkpoint를 관리하고 selected best를 load한 뒤 fit-end hook을 호출한다. | 확인된 사실 | `CVB:src/core/engine.py#Trainer.fit` | 04 §6 |
| CVB checkpoint는 model/optimizer/scheduler/scaler/RNG/config/env를 저장한다. | 확인된 사실 | `CVB:src/core/checkpoint.py#save_checkpoint`, `#load_checkpoint` | 04 §6, 07 §5 |
| memory bank, auxiliary stream, calibration hook의 정확한 위치는 model별 pinned source audit 전 미결정이다. | 미결정 | `P2-T01`, `P3-T01`, `P4-T02` | 03 §6, 04 §7 |

## 6. output, metric과 visualization

| 판단 | 분류 | 근거 | 연결 문서 |
|---|---|---|---|
| DV evaluator는 `pred_score`/label로 image metric, `anomaly_map`/mask로 pixel metric을 계산한다. | 확인된 사실 | `DV:src/defectvad/common/evaluator.py#Evaluator` | 05 §4 |
| DV predict path는 optional train-loader calibration과 visualizer option을 제공한다. | 확인된 사실 | `DV:experiments/predict.py#predict`, `src/defectvad/common/visualizer.py#Visualizer` | 05 §4 |
| ROI evaluator는 final corner prediction을 scalar `metrics.json`으로, predictor는 sample result를 CSV로 저장한다. | 확인된 사실 | `ROI:src/core/evaluator.py#Evaluator`, `src/core/predictor.py#Predictor` | 05 §5 |
| CVB adapter owns metric update/compute and optional prediction saving/visualization. | 확인된 사실 | `CVB:src/core/adapter.py#TaskAdapter`, `src/cli/commands.py#evaluate` | 05 §6 |
| anomaly map storage, threshold source 및 reference tolerance는 approved protocol 전 미결정이다. | 미결정 | `P0-T02`, `P0-T05`, SPEC §11 | 05 §7~§8 |

## 7. platform과 orchestration

| 판단 | 분류 | 근거 | 연결 문서 |
|---|---|---|---|
| ROI `BASE` + `CONFIGS`가 model/network/head/dataset condition을 compose한다. | 확인된 사실 | `ROI:scripts/batch_config.py#BASE`, `#CONFIGS` | 06 §5 |
| CVB benchmark runner는 existing output 보호, optional split filter, failure row 수집을 제공한다. | 확인된 사실 | `CVB:src/bench/runner.py#execute_split`, `#run_benchmark` | 06 §6 |
| CVB registry는 duplicate/unknown name error와 namespace별 build를 제공한다. | 확인된 사실 | `CVB:src/core/registry.py#Registry` | 07 §4 |
| CVB config validation은 schema, registry, path, monitor와 device를 확인한다. | 확인된 사실 | `CVB:src/core/config.py#validate_config` | 07 §4 |
| CVB offline guard는 non-loopback network/DNS를 차단하고 local weight failure를 명시한다. | 확인된 사실 | `CVB:src/core/offline.py#enable_offline_guard`, `#load_local_weights` | 07 §6 |
| inventory schema, checksum policy 및 model source approval workflow는 미결정이다. | 미결정 | `P0-T03`, `P0-T04`, 07 §7 | 07, 08 |

## 8. migration 판단과 후속 검증

| migration 항목 | 현재 판정 | 근거/후속 gate | 연결 문서 |
|---|---|---|---|
| pure-PyTorch model algorithm | 조정 | pinned source diff와 reference parity | 03 §7, 08 §3 |
| common trainer/checkpoint/context | 재사용 또는 대체 | `P1` contract fixture, resume test | 04 §6, 07 §5, 08 §3 |
| legacy common evaluator/visualizer | 대체 또는 조정 | explicit adapter output protocol | 05 §4, 08 §3 |
| ROI CLI/batch usability | 재사용 또는 조정 | manifest/result artifact/failure semantics | 06 §5~§7, 08 §3 |
| central model-name factory branch | 제외 | registry-based extension | 03 §5, 07 §4, 08 §3 |
| cross-model common abstraction | 미결정 | two-model evidence before promotion | 03 §7, 08 §4 |

## 9. 색인 유지 규칙

1. 비교 문서의 주요 사실을 수정할 때 이 표의 revision, file 또는 symbol을 같은 변경에서 갱신한다.
2. current checkout이 바뀌거나 line/symbol이 사라지면 기존 결론을 자동 승계하지 않고 해당 행을 재검증한다.
3. `확인된 사실` 행에는 code/config 근거를, `권고`와 `미결정` 행에는 상위 문서 또는 승인 task를 연결한다.
4. 이 색인은 source code의 전체 목록이 아니며, implementation 전 source audit과 license/protocol 검토를 대체하지 않는다.

작성일: 2026-08-20  
상태: 01~08 비교 문서의 주요 판단 역방향 색인 초안
