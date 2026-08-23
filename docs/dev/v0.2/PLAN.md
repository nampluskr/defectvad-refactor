# PLAN — 남은 anomalib 모델 포팅 우선순위

문서 상태: Draft Plan (제안, 미확정)
상위 문서: `BRIEF.md`
관련 문서: `docs/guides/anomaly-models.md` (모델 구현 현황), `.claude/skills/add-anomalib-model/SKILL.md` (포팅 절차)
작성일: 2026-08-23

## 1. 배경

`docs/guides/anomaly-models.md` 기준으로 현재 6개 모델(STFPM, EfficientAD, FastFlow, PatchCore, PaDiM, Reverse Distillation)이 구현되어 있다. 이 문서는 이후 어떤 anomalib 모델을 어떤 순서로 포팅할지에 대한 제안이며, 아직 사용자 승인을 받지 않았다.

## 2. 우선순위 판단 기준

1. **오프라인 원칙 준수** — `BRIEF.md` 원칙3(로컬 자산만 사용)을 위반하지 않는가. 인터넷 접근이나 외부 API가 필요한 모델(WinCLIP, VLM-AD 등)은 후순위로 미룬다.
2. **패러다임 다양성** — `docs/guides/anomaly-models.md` §2의 분류 체계에서 아직 비어 있는 패러다임(Reconstruction, Density Estimation 등)을 채우는가.
3. **포팅 비용** — 기존 컴포넌트(`components/feature_extractors`, `FrEIA` 등)를 재사용해 낮은 비용으로 포팅 가능한가, 아니면 새로운 학습 구조(2단계 학습, adversarial loss, backbone fine-tuning 등)가 필요한가.

## 3. 구현 완료 (참고, 재작업 대상 아님)

| 패러다임 | 모델 |
|---|---|
| Teacher–Student / Knowledge Distillation | STFPM, EfficientAD, Reverse Distillation |
| Normalizing Flow | FastFlow |
| Feature Embedding / Memory Bank | PatchCore, PaDiM |
| Density Estimation | DFM, DFKDE |
| Feature Embedding / Memory Bank (gradient 학습) | CFA |

DFM·DFKDE·CFA는 코드 포팅과 registry 등록, config 작성, offline 팩토리 구성을 완료했고, 반대 벤더(Codex CLI) 적대적 검증을 1회 거쳤다(`docs/dev/v0.2/reviews/A1.md`). CFA는 사용자가 실제로 `scripts/train.py`(bottle)를 실행해 학습이 정상 완주됨을 확인했다 — 이 과정에서 적대적 검토가 잡아내지 못한 결함 2건(YAML `1e-5` 파싱 함정, `CfaLoss.radius`의 non-leaf 텐서 재사용으로 인한 2스텝째 backward 실패)이 드러나 수정했다(`docs/dev/v0.2/reports/UPSTREAM-INVENTORY.md` §12.4). 이후 사용자가 DFM·DFKDE·CFA 세 모델 모두 train/evaluate/predict를 직접 실행해 정상 동작을 확인했다(2026-08-23) — 구체적인 image/pixel AUROC 수치는 별도로 기록되지 않았다. 3개 카테고리(bottle·carpet·capsule) 기준 정식 성능 비교는 여전히 사용자 실행 대기 상태다. 상세 구현 개요는 `docs/guides/anomaly-models.md` §3.7~§3.9를 참조한다.

CFA는 착수 전 우려했던 "backbone fine-tune으로 SSOT 전제가 흔들리는 사례"가 실제로는 발생하지 않았다: 모델 원본은 backbone을 얼린 채로 두지 않지만(anomalib 자체가 `model.parameters()` 전체를 optimizer에 넘김), forward가 항상 `torch.no_grad()`로 backbone을 실행하므로 그레이디언트가 도달하지 않는다. 이 프로젝트는 factory에서 backbone을 `requires_grad=False`로 명시적으로 고정해 다른 모델과 동일한 "backbone 고정" 관례를 유지했다 — 결과는 anomalib과 동일하다.

## 4. 우선순위 제안표 (DFM · DFKDE · CFA 제외 — 구현 완료, §3 참조)

| 순위 | 모델 | 채우는 패러다임 | 근거 | 리스크 · 확인 필요 사항 |
|---|---|---|---|---|
| 1 | CFLOW | Normalizing Flow (변형) | FastFlow와 동일 계열, `FrEIA` 의존 이미 확보 | FastFlow와 구조적 차별점이 세부사항 위주 |
| 2 | DRAEM | Reconstruction (discriminative) | Teacher-Student가 아닌 순수 reconstruction+discriminative 조합 | synthetic anomaly 생성용 texture 데이터셋(예: DTD)이 로컬 자산으로 추가 필요 — 원칙3 충족 여부 확인 |
| 3 | GANomaly | Reconstruction (adversarial) | GAN 기반 reconstruction, generator+discriminator 이원 최적화 사례 확보 | optimizer 2개, adversarial loss로 adapter 복잡도 상승 |
| 4 | CS-Flow | Normalizing Flow (cross-scale) | CFLOW/FastFlow와 비교되는 multi-scale flow 사례 | 우선순위 1·3 이후 여력 있을 때 |
| 5 | DSR | Reconstruction (discrete latent) | discrete codebook 기반, 구조 이질적 | 2단계 학습(사전학습 codebook + 본학습)이 engine의 단일 학습 루프 가정과 마찰 가능 |
| 보류 | WinCLIP, VLM-AD 등 | Vision-Language / Zero-shot | — | 학습→평가→추론 파이프라인 가정과 근본적으로 다름, 오프라인 원칙 저촉 가능성 — 별도 설계 검토 없이는 보류 |

## 5. 미정 사항

- 남은 순위 1~5(CFLOW·DRAEM·GANomaly·CS-Flow·DSR)의 확정 여부 — 사용자 승인 대기.
- DRAEM 착수 전 원칙3(오프라인) 저촉 여부 재검토 필요 — synthetic anomaly 생성용 texture 데이터셋 로컬화 방법 확정 (표의 "리스크" 열 참조).
- DFM·DFKDE·CFA는 세 스크립트(train/evaluate/predict) 실행 자체는 사용자가 확인했다(2026-08-23). 3개 카테고리(bottle·carpet·capsule) 기준 정식 성능 비교와 수치 기록은 아직 없다 — 필요 시 결과에 따라 config 하이퍼파라미터(특히 CFA `train.epochs`, DFM `score_type`)를 조정할 수 있다.
- PatchCore·PaDiM·DFM·DFKDE의 `runtime.amp: true` 비호환 가능성, `weights_path=None` 시 silent random-init — 적대적 검토(A1)에서 지적됐으나 9개 모델에 걸친 기존 설계라 이번 세션에서는 수정하지 않았다. 별도 과제로 core 변경 필요 (`UPSTREAM-INVENTORY.md` §14).
- 각 모델 착수 시 `.claude/skills/add-anomalib-model/SKILL.md` 절차를 그대로 따르되, 이 문서의 순서를 갱신한다.

## 6. 문서 갱신 규칙

모델을 포팅하면:
1. 이 표에서 해당 행을 "구현 완료" 표(§3)로 옮긴다.
2. `docs/guides/anomaly-models.md`를 `/add-anomalib-model` 절차에 따라 갱신한다(§7 문서 갱신 기준 참조).
