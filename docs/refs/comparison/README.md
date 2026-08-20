# `defectvad`, `roi-corner-detection-ver3`, `cv_boilerplate` 비교 분석

## 1. 목적

이 폴더는 사용자가 bottom-up 방식으로 작성한 레거시 `defectvad`, 사용자가 CLI와 batch process를 직접 변경하고 운용할 수 있는 대규모 실험 코드 `roi-corner-detection-ver3`, AI 에이전트가 상위 구조를 일괄 작성한 `cv_boilerplate`를 코드 근거에 따라 비교한다. 단순히 현재 구조와 기능의 차이를 나열하는 데 그치지 않고, 구체적인 anomaly model 요구와 사용자가 이해하는 실험 orchestration 방식이 `cv_boilerplate`의 공통 책임, 계층 및 extension point와 어떻게 대응되는지를 추적한다.

`roi-corner-detection-ver3`에서는 model, network, head, dataset과 실행 조건을 CLI 및 Python batch config로 조립하고 train, evaluate, predict를 반복 실행하는 방식을 중점적으로 살핀다. 이 저장소는 anomaly architecture의 기준이 아니라 사용자가 이미 자유롭게 수정하고 실행할 수 있는 workflow와 batch 사용성의 참고 구현이다.

또한 `defectvad`와 `roi-corner-detection-ver3`에 없었지만 `cv_boilerplate`에 도입된 registry 등의 설계 방식을 식별하고, 각 방식이 해결하려는 문제와 실제 적용 구조 및 anomaly detection 통합에 미치는 영향을 분석한다. 코드와 이력만으로 도입 의도를 확정할 수 없는 경우에는 확인된 구조적 효과와 해석을 구분하고 미결정 사항으로 남긴다.

이 비교 분석의 최종 목적은 `defectvad`의 anomaly detection 기능을 `cv_boilerplate` 방식으로 통합하고 가용하다고 판정된 모든 SOTA anomaly detection 모델을 지속적으로 포함할 수 있게 하는 것이다. 동시에 사용자가 향후 모델 개발, model 조건, dataset, 실험 구성과 평가 조건 변경 및 CLI와 batch benchmark 실행을 AI 에이전트에게 의존하지 않고 코드 수준에서 직접 수행할 수 있어야 한다. 문서는 어떤 파일과 symbol을 왜 변경해야 하는지, 변경한 구성요소가 다른 계층과 어떻게 연결되는지, 동일한 구성을 CLI, notebook과 batch process에서 어떻게 조립해 실행하는지를 사용자가 스스로 판단할 수 있는 수준으로 설명한다.

비교 분석의 핵심 질문은 다음과 같다.

1. `defectvad`의 bottom-up 구현은 구체적인 문제와 모델별 요구를 어떤 코드 구조로 해결했는가.
2. `roi-corner-detection-ver3`는 다양한 model, network, head, 조건과 dataset을 CLI 및 batch process로 어떻게 조립하고 실행하는가.
3. 두 사용자 구현에서 확인된 요구와 사용성이 `cv_boilerplate`의 어떤 상위 계층, 공통 책임 및 extension point로 확장 또는 일반화될 수 있는가.
4. 세 저장소는 train, evaluate, predict와 benchmark lifecycle 및 각 계층의 책임을 어떻게 나누는가.
5. Registry처럼 사용자 구현에 없거나 다르게 표현된 방식은 어떤 문제를 해결하며 `cv_boilerplate`에서 어떻게 동작하는가.
6. `roi-corner-detection-ver3`의 CLI와 batch 방식 중 사용자가 자유롭게 조건을 바꾸는 경험을 보존할 요소와 anomaly workflow에 맞게 조정할 요소는 무엇인가.
7. `defectvad`의 기능 중 사용자 의도를 충족하기 위해 보존해야 하는 것은 무엇이며, `cv_boilerplate`가 이미 대체한 책임은 무엇인가.
8. 모델별 차이를 최소한의 adapter로 수용하려면 어떤 gap을 해결해야 하며, 어떤 코드를 재사용, 조정, 대체, 제외 또는 미결정으로 판정해야 하는가.
9. 사용자가 새 모델이나 실험·평가 조건을 직접 추가 또는 변경하려면 어떤 파일, contract, 등록 지점 및 CLI, notebook, batch 실행 경로를 이해해야 하는가.
10. 가용한 SOTA anomaly model inventory를 확장하면서 source, license, local asset, lifecycle과 reference protocol을 어떻게 추적할 것인가.

## 2. 문서 경계

비교 분석 결과는 `docs/refs/comparison/` 내부에 기록한다. 상위의 기존 요구사항과 설계 문서는 비교 과정에서 읽기 전용 기준으로 참조하며 이 문서군을 작성하기 위해 내용을 수정하지 않는다. 단일 탐색 진입점 원칙을 지키기 위해 `WIKI_INDEX.md`에는 비교 문서의 탐색 정보와 상태만 동기화한다.

| 대상 | 역할 | 변경 정책 |
|---|---|---|
| `docs/refs/BRIEF.md` | 사용자 의도, 방향, 범위의 기준 | 수정하지 않음 |
| `docs/refs/PRD.md` | 기능, 품질, 제약 및 완료 요구사항의 기준 | 수정하지 않음 |
| `docs/refs/SPEC.md` | 현재 구조 분석과 목표 기술 설계의 기준 | 수정하지 않음 |
| `docs/refs/PLAN.md` | 단계별 구현 계획의 기준 | 수정하지 않음 |
| `docs/refs/backlog.json` | 작업과 검증 항목의 실행 색인 | 수정하지 않음 |
| `docs/refs/WIKI_INDEX.md` | 기존 문서와 비교 문서의 탐색 색인 | 탐색 정보와 상태만 동기화 |
| `docs/refs/comparison/` | 코드 비교 결과, 근거, gap 및 권고 | 이 폴더 안에서만 작성 |

비교 결과가 기존 문서와 충돌하거나 기존 문서에 없는 gap을 드러내더라도 상위 요구사항과 설계 문서를 직접 고치지 않는다. 해당 사실과 영향, 가능한 대안, 미결정 사항을 비교 문서에 남기고 `WIKI_INDEX.md`에는 탐색 경로만 반영한다.

## 3. 분석 대상과 기준 revision

분석 기준일은 2026-08-20이다.

| 구분 | 경로 | Branch | Revision | 확인 상태 |
|---|---|---|---|---|
| 현재 구조 | `D:\_clones\cv_boilerplate` | `main` | `65d5412b0fa29ec817cfffc94ccfc177a4d9aad5` | 작업 트리 변경 없음 |
| 레거시 근거 | `D:\_clones\defectvad` | `main` | `14879ea2a8970cee25438500e5abfeeb4be8e358` | 작업 트리 변경 없음 |
| 사용자 운용 workflow 근거 | `D:\_clones\roi-corner-detection-ver3` | `main` | `8ae989a88996441e44fb2d5296a6419a8f661220` | 작업 트리 변경 없음 |
| 문서 작성 대상 | `D:\_clones\defectvad-refactoring` | 해당 없음 | Git 메타데이터 없음 | `comparison/`만 변경 |

기존 문서가 분석한 공개 `cv_boilerplate` 기준 revision은 `71261cef`이고 현재 checkout은 `65d5412b`이다. 두 revision의 차이가 아직 완전히 대조되지 않았으므로, 기존 문서의 설명과 현재 코드가 다를 경우 이를 곧바로 오류로 단정하지 않고 revision 차이의 가능성을 함께 기록한다.

revision이 바뀌면 기존 결론을 그대로 승계하지 않는다. 변경된 파일과 영향을 받는 symbol을 먼저 확인하고 관련 비교 항목을 다시 검증한다.

## 4. 저장소의 역할

### 4.1 `defectvad`

`defectvad`는 레거시 요구사항, 모델별 lifecycle, 과거 구현 선택과 시행착오를 확인하기 위한 증거다. 레거시에 존재한다는 이유만으로 해당 구조를 유지 대상으로 판단하지 않는다.

### 4.2 `cv_boilerplate`

`cv_boilerplate`는 공통 train, evaluate, predict, benchmark lifecycle과 현재 extension point를 확인하기 위한 기준 구현이다. anomaly-specific 또는 model-specific 책임이 공통 engine과 CLI에 침투하지 않는지를 중점적으로 살핀다.

### 4.3 `roi-corner-detection-ver3`

`roi-corner-detection-ver3`는 다양한 model, network, head, dataset과 실행 조건을 사용자가 직접 바꾸는 CLI, Python batch config, subprocess orchestration 및 결과 수집 방식을 확인하기 위한 근거다. ROI corner detection의 task-specific target, head와 geometry를 anomaly architecture로 이전하지 않으며, 사용자가 이해하는 조립 방식과 대규모 비교 workflow를 선별해 참고한다.

### 4.4 기존 v0.1 문서

기존 문서는 사용자 의도와 목표 설계의 기준이다. 비교 문서는 기존 문서의 내용을 복제하지 않고 코드가 그 의도와 설계를 어떻게 지원하거나 제약하는지에 대한 근거를 제공한다.

## 5. 비교 범위

다음 항목을 비교한다.

- 저장소와 package 구조, 실행 진입점 및 의존 방향
- config loading, object construction, factory, registry와 adapter
- train, evaluate, predict와 benchmark lifecycle
- model, network, head, dataset 및 실행 조건의 조합 표현
- CLI option, batch config, subprocess 실행, 실패 처리와 결과 수집
- dataset, split, transform, batch와 model output contract
- loss, optimizer, scheduler와 모델별 학습 단계
- fitting, auxiliary data, calibration과 post-processing
- metric 계산과 anomaly score 및 anomaly map 의미
- checkpoint, model state, resume와 결과 선택
- local asset, pretrained weight와 offline 동작
- logging, 오류 처리, 시각화와 실행 결과물
- unit, integration, smoke 및 reference benchmark 검증 가능성
- 모델별 통합 난이도와 공통 abstraction의 한계

다음 작업은 이 비교 문서군의 범위에 포함하지 않는다.

- 제품 코드 구현 또는 리팩터링
- 학습, 평가, 추론 또는 장시간 benchmark 실행
- dependency 설치나 변경
- dataset 또는 pretrained weight 다운로드
- `defectvad`, `cv_boilerplate`와 `roi-corner-detection-ver3`의 파일 수정
- anomalib 전체 구현에 대한 독립적인 재설계

## 6. 문서 구성

| 순서 | 문서 | 내용 | 상태 |
|---:|---|---|---|
| 0 | `README.md` | 범위, 기준 revision, 탐색 순서와 작성 규칙 | 현재 문서 |
| 1 | [01_COMPARISON_OVERVIEW.md](01_COMPARISON_OVERVIEW.md) | 전체 실행 흐름, 사용자 운용 구조와 상위 구조의 대응 | 세 저장소 비교 초안 |
| 2 | [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md) | dataset, transform, split, collate, dataloader와 batch data condition | 세 저장소 비교 초안 |
| 3 | `03_MODEL_AND_ADAPTER.md` | model, wrapper, adapter, loss와 optimizer 경계 | 예정 |
| 4 | `04_EXECUTION_LIFECYCLE.md` | trainer, evaluator와 predictor의 train/evaluate/predict 흐름 | 예정 |
| 5 | `05_OUTPUT_AND_VISUALIZATION.md` | output, metric, post-processing, threshold와 visualizer | 예정 |
| 6 | `06_CLI_AND_BATCH_ORCHESTRATION.md` | CLI, notebook 조립 경로, 반복 실행과 benchmark orchestration | 예정 |
| 7 | `07_PLATFORM_MECHANISMS.md` | registry, factory/builder, config, checkpoint, context, logging, error와 offline | 예정 |
| 8 | `08_MIGRATION_SUMMARY.md` | 변경·일반화 요약, 효과, 제약, gap과 이전 판정 | 예정 |
| 9 | `09_EVIDENCE_INDEX.md` | 주요 판단과 코드 근거의 역방향 색인 | 예정 |

권장 탐색 순서는 문서 번호와 같다. 특정 결론의 코드 근거만 찾을 때는 향후 작성할 `09_EVIDENCE_INDEX.md`에서 저장소, 파일 및 symbol을 기준으로 역추적한다.

## 7. 판단 기준

### 7.1 우선순위

문서와 구현의 의미가 충돌할 때 다음 우선순위를 따른다.

```text
사용자가 명시적으로 수정한 의도
    > BRIEF.md
    > PRD.md
    > SPEC.md
    > PLAN.md
    > backlog.json
    > 현재 cv_boilerplate 구현
    > roi-corner-detection-ver3의 사용자 운용 구현
    > defectvad 레거시 구현
```

코드에서 다른 사실이 확인되면 상위 의도를 임의로 변경하지 않고 충돌과 영향을 비교 문서에 기록한다.

### 7.2 레거시 기능 판정

`defectvad`의 기능을 이전 대상으로 제안하기 전에 다음을 확인한다.

1. 현재 사용자 의도에 필요한가.
2. `cv_boilerplate`가 이미 같은 책임을 담당하는가.
3. anomaly task 또는 특정 model에 속하는 책임인가.
4. copy-and-paste 또는 bottom-up 개발 과정에서 생긴 우연한 구조인가.
5. anomalib reference 성능 재현에 실제로 필요한가.

### 7.3 사용자 운용 workflow 판정

`roi-corner-detection-ver3`의 방식을 참고하기 전에 다음을 확인한다.

1. 사용자가 직접 조건을 바꾸고 실행하는 데 실제로 도움이 되는가.
2. `cv_boilerplate`의 config, registry, CLI 또는 benchmark extension point로 표현할 수 있는가.
3. ROI corner task에만 유효한 model, head, target 또는 dataset 가정을 anomaly 공통 책임으로 옮기지 않는가.
4. 여러 model과 조건의 조합, 실패 격리 및 결과 비교를 더 명시적으로 만드는가.
5. 단일 실행과 batch 실행이 같은 resolved condition과 lifecycle을 사용하는가.

### 7.4 abstraction 판정

새 abstraction을 권고하기 전에 다음을 확인한다.

1. 기존 extension point로 해결할 수 있는가.
2. 최소 adapter로 해결할 수 있는가.
3. 둘 이상의 모델에서 공통 필요성이 확인되는가.
4. 공통 engine에 task명 또는 model명 분기를 요구하지 않는가.
5. upstream 알고리즘 보존과 reference 검증을 방해하지 않는가.

## 8. 근거와 서술 규칙

### 8.1 구성요소별 설명 순서

비교 문서는 저장소별 기능 목록을 각각 나열하는 방식보다 구성요소별 변화 과정을 중심으로 작성한다. 각 구성요소는 가능한 한 다음 순서를 따른다.

1. 구성요소가 해결하는 문제와 비교 범위를 짧게 설명한다.
2. `defectvad`에서 사용자가 구현한 클래스와 함수의 스켈레톤, 호출 관계 및 주요 코드 구현 방식을 제시한다.
3. 해당 구현이 구체적인 모델 또는 실행 문제를 어떻게 해결했는지 설명한다.
4. `roi-corner-detection-ver3`에 대응 기능이 있으면 CLI, batch config, 조립과 결과 수집 방식을 같은 추상화 수준에서 제시한다.
5. `cv_boilerplate`에서 대응 책임이 어느 계층, contract 또는 extension point에 있는지 제시한다.
6. 세 구현의 호출 흐름과 핵심 구조가 어떻게 다르며 무엇을 목표 workflow에 채택할지 설명한다.
7. Registry처럼 직접 대응물이 없는 방식은 해결하려는 문제, 적용 구조, 등록부터 조회와 생성까지의 동작 순서로 설명한다.
8. 얻는 효과와 함께 복잡성, 제약, 잃어버린 특성 또는 anomaly integration에 남은 gap을 기록한다.
9. 사용자가 모델이나 실험·평가 조건을 직접 바꿀 때 수정할 파일과 symbol, 등록 또는 설정 지점, 영향 범위 및 CLI, notebook, batch 실행 경로를 정리한다.

클래스나 함수의 전체 구현을 그대로 복사하지 않는다. 사용자가 구조와 차이를 이해하는 데 필요한 최소 스켈레톤과 핵심 구문만 인용하고, 생략한 부분은 주석 또는 설명으로 명시한다. 세 저장소의 예시는 같은 추상화 수준과 같은 실행 단계가 나란히 보이도록 배치한다. 직접 대응물이 없으면 그 사실과 비교하지 않는 이유를 명시한다.

각 구성요소의 권장 본문 형식은 다음과 같다.

```text
구성요소와 해결 문제
  -> defectvad의 스켈레톤과 주요 구현
  -> defectvad에서의 실제 호출 및 책임
  -> roi-corner-detection-ver3의 대응 조립 및 사용자 운용 방식
  -> cv_boilerplate의 대응 스켈레톤과 주요 구현
  -> 세 구현에서 채택, 대체 또는 일반화할 지점
  -> 변경 이유에 대한 근거와 해석
  -> 효과, 제약, 통합 gap 및 미결정 사항
  -> 사용자가 직접 변경할 지점과 CLI/notebook 실행 경로
```

코드가 직접 보여 주는 구조와 동작은 `확인된 사실`로 기록한다. 변경 이유는 commit, 문서 또는 명시적인 사용자 설명으로 확인된 경우에만 사실로 서술하며, 구조적 효과를 바탕으로 추론한 이유는 `해석`으로 구분한다.

### 8.2 근거 표기와 비교 수준

비교 문서의 주요 문장은 다음 네 종류로 구분한다.

| 표기 | 의미 |
|---|---|
| `확인된 사실` | 지정된 revision의 코드나 기존 기준 문서에서 직접 확인한 내용 |
| `해석` | 둘 이상의 사실을 연결해 도출한 의미 또는 영향 |
| `권고` | 목표 구조와 제약을 고려한 선택 제안 |
| `미결정` | 코드만으로 확정할 수 없거나 사용자 결정이 필요한 항목 |

확인된 사실에는 가능한 한 다음 형식의 근거를 붙인다.

```text
저장소@revision:상대/파일/경로.py#SymbolName
```

symbol이 없거나 특정 구문이 핵심인 경우 line을 함께 기록한다. line 번호는 revision에 종속되므로 revision 없이 단독으로 인용하지 않는다.

비교표에서는 같은 추상화 수준의 항목만 나란히 둔다. 한 저장소의 파일과 다른 저장소의 전체 subsystem을 직접 대응시키지 않으며, 대응물이 없으면 `없음`, `다른 계층이 담당`, `비교 대상 아님`, `확인 필요` 중 하나로 명시한다.

## 9. 이전 판정 용어

기능과 모듈의 이전 방향은 다음 용어를 사용한다.

| 판정 | 의미 |
|---|---|
| 재사용 | 의미와 책임을 거의 바꾸지 않고 사용할 수 있음 |
| 조정 | 핵심 의미는 유지하되 현재 contract에 맞춘 adapter나 제한된 변경이 필요함 |
| 대체 | 같은 목적을 `cv_boilerplate`의 기존 기능이 더 적절하게 담당함 |
| 제외 | 현재 사용자 의도에 불필요하거나 우연한 레거시 구조이므로 이전하지 않음 |
| 미결정 | 근거 또는 사용자 결정이 부족하여 아직 판정할 수 없음 |

판정에는 반드시 근거, 목표 위치, 예상 영향과 검증 방법을 함께 기록한다. 단순한 파일명 유사성만으로 대응 관계나 재사용 가능성을 확정하지 않는다.

## 10. 완료 기준

각 비교 문서는 다음 조건을 충족해야 완료로 본다.

- 비교한 저장소와 revision이 명시되어 있다.
- 세 저장소의 architecture, anomaly algorithm 근거 및 사용자 운용 workflow 역할이 구분되어 있다.
- 확인된 사실과 해석, 권고 및 미결정 사항이 구분되어 있다.
- 주요 판단에 파일과 symbol 수준의 근거가 있다.
- 기존 요구사항 식별자를 변경하지 않고 필요한 경우 참조만 한다.
- 기존 문서와의 충돌 또는 revision 차이가 숨겨지지 않는다.
- `defectvad`의 구조를 목표 architecture로 전제하지 않는다.
- `cv_boilerplate`의 기존 extension point를 먼저 검토한다.
- `roi-corner-detection-ver3`의 task-specific 구조를 그대로 목표 architecture로 전제하지 않는다.
- 사용자가 AI 에이전트 없이 모델, head에 대응하는 model option, dataset과 실험·평가 조건의 변경 지점 및 영향 범위를 찾을 수 있다.
- 동일한 구성요소를 CLI, notebook과 batch process에서 조립해 실행하는 경로가 식별되어 있다.
- 가용한 SOTA anomaly model의 source, license, local asset, lifecycle과 reference protocol을 추적할 확장 방향이 식별되어 있다.
- 문서가 UTF-8로 저장되고 한글이 보존된다.
- Unicode replacement character와 이모지가 없다.

작성일: 2026-08-20  
상태: 3개 저장소 비교 기준 및 01·02 초안 반영
