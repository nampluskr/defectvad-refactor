# `defectvad`, `roi-corner-detection-ver3`, `cv_boilerplate`의 CLI와 batch orchestration 비교

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

이 문서는 사용자가 model, dataset, 실행 조건을 변경해 train, evaluate, predict 및 반복 benchmark를 실행하는 경로를 비교한다. 목표는 ROI 프로젝트에서 확인된 명시적 조립과 실패 격리 사용성을 `cv_boilerplate`의 config, registry 및 benchmark runner 위에 표현하는 것이다.

Model/adapter의 책임은 [03_MODEL_AND_ADAPTER.md](03_MODEL_AND_ADAPTER.md), lifecycle은 [04_EXECUTION_LIFECYCLE.md](04_EXECUTION_LIFECYCLE.md), 결과 artifact는 [05_OUTPUT_AND_VISUALIZATION.md](05_OUTPUT_AND_VISUALIZATION.md)에서 다룬다.

## 2. 분석 기준

분석 기준일은 2026-08-20이다. 정적 script, config와 CLI parser만 조사했으며 subprocess, train 또는 benchmark는 실행하지 않았다.

| 저장소 | Revision | 사용자 운용 비교에서의 역할 |
|---|---|---|
| `defectvad` | `14879ea2a8970cee25438500e5abfeeb4be8e358` | Python 상수 목록으로 model/category를 반복 실행한 레거시 근거 |
| `roi-corner-detection-ver3` | `8ae989a88996441e44fb2d5296a6419a8f661220` | CLI option과 Python batch config로 조건을 조립한 사용성 근거 |
| `cv_boilerplate` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` | config override와 failure-isolated benchmark runner의 현재 근거 |

관련 요구사항은 `FR-001`, `FR-018`~`FR-020`, `FR-022`, `FR-024`, `FR-025`, `NFR-002`, `NFR-008`, `NFR-010`, `CON-006`, `CON-007`, `CON-009`, `CON-010`이며, 관련 설계와 계획은 SPEC §11, §15 및 `P0-T02`, `P5-T01`~`P5-T05`다.

## 3. 한눈에 보는 결론

```text
defectvad
run_training.py의 Python 상수
  -> dataset/category/model 조합
  -> train.py subprocess
  -> 첫 실패에서 return

roi-corner-detection-ver3
공통 CLI + Python BASE/CONFIGS
  -> model/network/head/dataset 조건
  -> train/evaluate/predict subprocess
  -> 실패를 기록하고 다음 조합 계속

cv_boilerplate
YAML + --set + benchmark manifest
  -> named split의 resolved config
  -> in-process command execution
  -> split별 실패 row와 leaderboard/control report
```

| 질문 | `defectvad` | `roi-corner-detection-ver3` | `cv_boilerplate` |
|---|---|---|---|
| 단일 실행 선택 | `--dataset`, `--category`, `--model` | model/network/head/dataset CLI option | config path와 repeated `--set` |
| 반복 조합 표현 | script 상단 Python list/boolean | `BASE`와 `CONFIGS` dict 목록 | benchmark manifest의 named split |
| train/evaluate/predict 연결 | 별도 runner script | 같은 config가 세 mode에 전달 | 같은 resolved config builder와 command |
| 실패 처리 | 첫 command failure 후 return | case failure를 수집한 후 계속 | split exception을 row로 수집한 후 계속 |
| 결과 집계 | terminal print 중심 | terminal summary 중심 | result rows, leaderboard와 control report |
| overwrite 보호 | 확인된 공통 보호 없음 | auto name 충돌 가능성 | output 존재 시 명시적 `--overwrite` 요구 |

`권고` anomaly inventory의 한 case는 Python code 주석 해제가 아니라 version-controlled declarative manifest로 표현한다. case마다 model/source revision, license, local asset, dataset/split, resolved overrides, lifecycle mode, checkpoint policy, metric/threshold protocol, expected artifact와 failure policy를 기록한다. 사용자는 CLI override로 빠르게 조건을 바꿀 수 있어야 하지만 acceptance run은 immutable manifest와 resolved config를 남겨야 한다.

## 4. `defectvad`: 직접 수정하는 Python 실행 목록

`확인된 사실` `run_training.py`, `run_evaluation.py`, `run_prediction.py`는 dataset list, category map, model list, epoch 수와 boolean option을 file-level Python 상수로 둔다. 각 조합에 대해 해당 script를 subprocess로 실행한다.

근거:

- `defectvad@14879ea2:experiments/run_training.py#run`
- `defectvad@14879ea2:experiments/run_evaluation.py#run`
- `defectvad@14879ea2:experiments/run_prediction.py#run`

`확인된 사실` 각 runner는 subprocess return code가 0이 아니면 오류 context를 print하고 즉시 `return`한다. train/evaluate/predict 목록과 option은 서로 다른 script에 중복되어 있으며, 현재 active case는 comment/uncomment와 상수 변경으로 선택한다.

`해석` 사용자가 최소한의 Python 수정으로 category와 model 조합을 바꾸는 방법은 명확하다. 그러나 동일 조건이 세 runner에 흩어져 train/evaluate/predict identity가 어긋날 위험이 있고, 한 실패가 뒤 case의 결과 수집까지 막는다. terminal output만으로는 batch 전체의 machine-readable status나 resolved protocol을 역추적하기 어렵다.

`이전 판정` 사람이 읽을 수 있는 조합 heading과 subprocess 단위 실행은 `조정`해 유지한다. Python 상수/주석으로 case를 활성화하고 첫 실패에서 중단하는 방식은 `대체`한다.

## 5. `roi-corner-detection-ver3`: 선택 축을 보이는 CLI와 config queue

`확인된 사실` 공통 parser는 `dataset`, `csv_path`, `model`, `network`, `head`, image/batch/epoch size, warm-up, worker, checkpoint와 output directory를 option으로 제공한다. `get_exp_name()`과 `get_output_dir()`는 model/network/head/dataset에서 기본 결과 경로를 만든다.

근거: `roi-corner-detection-ver3@8ae989a8:scripts/config.py#DEFAULTS`, `#get_exp_name`, `#get_output_dir`, `#parse_args`

`확인된 사실` `batch_config.py`는 공유 `BASE` dict에 model/network/head 및 case별 option을 덮어쓴 `CONFIGS`를 정의한다. `batch_run.py`는 `--mode train|evaluate|predict|all`, optional `--config`를 받고 config를 import한 뒤 case마다 subprocess를 만든다. train/evaluate/predict는 같은 `PASS_KEYS`와 config를 사용하며 checkpoint가 없으면 output directory의 `model.pth`를 evaluate/predict에 전달한다.

근거:

- `roi-corner-detection-ver3@8ae989a8:scripts/batch_config.py#BASE`, `#CONFIGS`
- `roi-corner-detection-ver3@8ae989a8:scripts/batch_run.py#load_configs`, `#get_cli_args`, `#run`

`확인된 사실` batch runner는 case별 `CalledProcessError`를 success/error record로 보관하고 다음 case를 계속 실행한다. 마지막에 실패 case를 출력하고 하나라도 실패하면 exit code 1을 반환한다. record는 process memory와 terminal summary에만 있고 summary artifact로 저장되지는 않는다.

`해석` 이 방식은 사용자가 조건 축과 실행 mode를 직접 보고 전체 비교를 통제하게 한다. 특히 one-case failure가 batch를 삭제하지 않는 점은 anomaly benchmark에 채택할 가치가 있다. 다만 Python module import로 config를 실행하고 `PASS_KEYS`가 parser의 전 option을 포괄하지 않는 점, 기본 experiment name이 모든 data/training identity를 반영하지 않는 점은 strict reference protocol의 canonical manifest로는 부족하다.

## 6. `cv_boilerplate`: resolved config와 failure-isolated benchmark

`확인된 사실` CLI parser는 config, train, evaluate, predict와 benchmark subcommand를 제공한다. 각 config-bound command는 repeated `--set` override를 받고 benchmark는 `--only`와 `--overwrite`도 받는다.

근거: `cv_boilerplate@65d5412b:src/cli/parser.py#build_parser`

`확인된 사실` benchmark runner는 manifest의 split config를 resolve한 뒤 split output directory가 이미 있으면 `--overwrite` 없이는 오류를 낸다. `run_benchmark()`은 optional `--only`로 requested split을 제한하고, split exception을 failed row로 보관한 뒤 나머지 split을 계속 실행한다. 이후 leaderboard와 control report를 생성하는 경로가 있다.

근거:

- `cv_boilerplate@65d5412b:src/bench/runner.py#execute_split`
- `cv_boilerplate@65d5412b:src/bench/runner.py#run_benchmark`
- `cv_boilerplate@65d5412b:src/bench/leaderboard.py#build_leaderboard`
- `cv_boilerplate@65d5412b:src/cli/commands.py#run_benchmark`

`해석` 현재 benchmark structure는 ROI batch runner의 failure isolation을 config-controlled, machine-readable comparison에 가깝게 옮긴 기반이다. 그러나 일반 vision task의 benchmark schema가 anomalib source/license/local asset/reference protocol까지 충분히 기술하는지는 이 비교만으로 확정할 수 없다. 해당 field는 inventory와 manifest 요구사항으로 명시적으로 추가 검토해야 한다.

## 7. 목표 anomaly case manifest

`권고` 다음 표는 목표 문서 schema이며 현 구현 file format을 확정하지 않는다.

| field 군 | 최소 내용 | 사용자 변경 지점 | 검증 |
|---|---|---|---|
| identity | case name, model, upstream revision, adapter version | inventory/benchmark case | duplicate 및 unknown registry name |
| data | dataset, category, materialized split, local paths | case override | path, checksum, leakage guard |
| lifecycle | train/evaluate/predict mode, epochs, checkpoint input/output | case override | selected checkpoint와 new-process round-trip |
| reference | source/license, environment, metric, threshold, tolerance | approved protocol section | required field and revision match |
| artifacts | run directory, metrics, prediction/map, log, provenance | runner-owned path | overwrite protection and existence |
| control | enabled, repeats, only filter, failure continuation | benchmark CLI | every requested case produces terminal row |

```text
case manifest -> resolve config + preflight
              -> one isolated run directory
              -> success / failure result row
              -> result artifact and provenance link
              -> aggregate leaderboard without hiding failures
```

`권고` CLI의 ad-hoc `--set`은 diagnosis와 exploration에 사용한다. 승인 benchmark는 case manifest를 기준으로 하며, CLI override가 있을 때는 override 전체를 copied resolved config와 result row에 남긴다. runner는 failure를 계속 수집하되, failed case를 leaderboard에서 성공 행처럼 취급하거나 final exit status를 0으로 바꾸지 않는다.

## 8. 미결정 사항

- train, evaluate, predict를 항상 한 benchmark case에서 연속 수행할지, approved checkpoint를 사용하는 evaluate-only/predict-only case를 어떻게 표현할지는 SPEC §11과 05 문서의 artifact contract를 함께 결정해야 한다.
- 병렬 execution, GPU resource scheduling, retry 및 timeout policy는 정적 조사로 정하지 않았다. reference repeat policy와 compute budget이 승인된 뒤 별도 설계가 필요하다.
- 기존 benchmark runner에 anomaly inventory field를 넣을지, inventory와 benchmark manifest를 별도 문서/파일로 연결할지는 07 및 08 문서의 platform/migration 판단에서 다룬다.

작성일: 2026-08-20  
상태: 세 저장소 정적 비교 초안
