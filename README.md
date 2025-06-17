# AI 역량 평가 프로젝트

본 프로젝트는 음성 기반 송금 시스템을 위한 자연어 이해(NLU) 모델 및 STT(Speech-to-Text) 성능 비교 분석을 포함합니다.

## 프로젝트 구조

```
client_ai_eval/
├── llms/                    # 자연어 이해(NLU) 모델 관련 파일
│   ├── main.py             # 전체 실행 진입점
│   ├── readme.md           # NLU 모듈 설명
│   └── model/
│       ├── runner.py       # 모델 로딩 및 실행 로직
│       ├── prompts.py      # 프롬프트 설계 전략
│       ├── configs.py      # 시스템 프롬프트 및 모델 설정
│       ├── QWEN_TEST.py    # 다양한 설정에서의 테스트 스크립트
│       └── results/        # 모델 실행 결과
│   └── labeled data/       # 실 데이터 기반 라벨링 데이터
│       ├── transfer.json
│       ├── non_memory.json
│       └── samples.json
├── STT/                    # 음성 인식(STT) 관련 파일
│   ├── benchmark.py        # Whisper 기반 STT 성능 측정
│   ├── distil_whisper_benchmark.py  # 경량 STT 모델 벤치마크
│   ├── whisper_cpp_results_all_ko.txt  # 한국어 STT 성능 결과
│   └── results.md          # 성능 기록 및 비교 리포트
```

## 주요 기능

### 1. 자연어 이해(NLU) 모델

- Qwen2.5 기반의 의도 분류 모델 구현
- 다양한 시스템 프롬프트에서의 테스트 결과 포함
- 실제 서비스 사용자 발화 이해 수준 평가
- Few-shot 학습 및 평가를 위한 라벨링 데이터 포함

### 2. 음성 인식(STT) 성능 분석

- Whisper 및 Whisper.cpp 기반 음성 인식 벤치마크
- 한국어에 대한 다양한 모델 성능 비교
- 경량화된 모델(Distil Whisper) 성능 평가

## 기술적 특징

- 모듈화된 구조로 기능별 명확한 분리
- 모델 로딩부터 실행, 결과 평가까지 일관된 파이프라인
- 실제 서비스 데이터 기반의 성능 평가
- 한국어 특화 성능 분석

## 사용 방법

각 모듈의 상세 사용 방법은 해당 디렉토리의 readme.md 파일을 참조하시기 바랍니다.
