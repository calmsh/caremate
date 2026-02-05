# 케어메이트 - AI 만성질환 예측 시스템

만성질환 위험도를 예측하고 AI 건강 상담을 제공하는 웹 애플리케이션입니다.

## 주요 기능
- 건강 정보 입력 및 분석
- 정신건강 설문 (PHQ-9, GAD-7, BP1, EQ5D)
- AI 기반 만성질환 위험도 예측
- 음성 지원 AI 건강 상담

## 설치 방법
```bash
pip install -r requirements.txt
```

## 실행 방법
```bash
streamlit run app4.py
```

## 필수 파일
- `health_models.pkl`: 학습된 머신러닝 모델 파일
- OpenAI API 키 설정 필요