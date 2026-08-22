# defectvad-refactor

`cv_boilerplate` 범용 컴퓨터 비전 실행 프레임워크 위에 anomalib 기반의 SOTA 비지도 이상 탐지(Anomaly Detection) 모델(STFPM, EfficientAD)을 pure-PyTorch로 통합하고, 유연한 Data/Model 분리 실행 환경을 제공하는 프로젝트입니다.

---

## 주요 문서 안내

### 사용자 및 아키텍처 가이드 (Guides)
- [프로젝트 폴더 구조 가이드](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/guides/structure.md): `configs/`, `scripts/`, `src/` 디렉터리 레이아웃 및 역할 정의
- [CLI 사용법 가이드](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/guides/cli-usage.md): 커맨드라인 명령어 실행 안내
- [가이드 목록](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/guides/README.md): 전체 가이드 문서 색인

### 개발 문서 (Development)
- [v0.2 BRIEF (AGY안)](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/docs/dev/v0.2/BRIEF-AGY.md): v0.2 실행 진입점 재설계, Data/Model 분리 및 단일/다중 평가 Use Case
- [AGENTS.md](file:///D:/projects/nampluskr/00_review/260820_defectvad-refactor/AGENTS.md): 저장소 개발 원칙 및 AI 에이전트 검증 규칙
