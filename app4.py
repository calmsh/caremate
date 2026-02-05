import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import io
import asyncio
import edge_tts  # 추가


# --- AI 및 음성 기능을 위한 라이브러리 추가 ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from streamlit_mic_recorder import speech_to_text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# [보안 설정 영역] - API 키를 여기에 입력하세요 (사용자에게는 보이지 않음)
# ---------------------------------------------------------
OPENAI_API_KEY = "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"

# --- 디자인 설정 ---
STYLE_CONFIG = {
 "corner_radius": "25px",      
 "border_width": "1px",
 "border_color": "#e2e8f0",
 "fg_color": "#FFFFFF",
 "bg_color": "#F0F9F6"
}

LEVEL_THEMES = {
 "높음": {"color": "#ef4444", "bg": "#fee2e2", "emoji": "🔴"},
 "중간": {"color": "#f59e0b", "bg": "#fef3c7", "emoji": "🟡"},
 "낮음": {"color": "#22c55e", "bg": "#dcfce7", "emoji": "🟢"}  
}

st.set_page_config(page_title="케어메이트 - AI 만성질환 예측", layout="centered", page_icon="🏥")

# --- 모델 로드 ---
@st.cache_resource
def load_models():
 if not os.path.exists('health_models.pkl'):
  st.error("❌ 모델 파일(health_models.pkl)이 없습니다. 먼저 모델을 학습시켜주세요.")
  st.stop()
 
 models = joblib.load('health_models.pkl')
 
 # 버전 호환성 패치
 for name, info in models.items():
  final_model = info['pipeline'].steps[-1][1]
  if 'LogisticRegression' in str(type(final_model)):
   if not hasattr(final_model, 'multi_class'):
    final_model.multi_class = 'ovr'
  if hasattr(final_model, 'estimators_'):
   for est in final_model.estimators_:
    actual_est = est.steps[-1][1] if hasattr(est, 'steps') else est
    if 'LogisticRegression' in str(type(actual_est)):
     if not hasattr(actual_est, 'multi_class'):
      actual_est.multi_class = 'ovr'
 
 return models

MODELS = load_models()

# --- 세션 상태 초기화 ---
if 'step' not in st.session_state:
 st.session_state.step = 1
if 'sub_step' not in st.session_state:
 st.session_state.sub_step = 1
if 'q_idx' not in st.session_state:
 st.session_state.q_idx = 0
if 'data_confirmed' not in st.session_state:
 st.session_state.data_confirmed = False
if 'user_data' not in st.session_state:
 st.session_state.user_data = {
  "name": "", "gender": "남성", "age": 70, "height": 160, "weight": 60,
  "diseases": [], "family_history": [], "edu": "대졸 이상", "marry": "기혼",
  "incm": "상", "alcohol": "아니오", "sleep_time": 7
 }
if 'survey_answers' not in st.session_state:
 st.session_state.survey_answers = {"PHQ9": {}, "GAD7": {}, "BP1": {}, "EQ5D": {}}
if 'chat_history' not in st.session_state:
 st.session_state.chat_history = []

# --- CSS 스타일 ---
st.markdown(f"""
<style>
 @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
 
 .stApp {{ background-color: {STYLE_CONFIG['bg_color']} !important; font-family: 'Noto Sans KR', sans-serif; }}
 .block-container {{ max-width: 700px !important; padding: 3rem 1rem !important; }}
 
 [data-testid="stVerticalBlock"] > div:has(div.card-content) {{
  background-color: white !important;
  padding: 40px !important;
  border-radius: {STYLE_CONFIG['corner_radius']} !important;
  border: {STYLE_CONFIG['border_width']} solid {STYLE_CONFIG['border_color']} !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important;
 }}
 
 .disease-item-card {{
  background-color: white;
  border-radius: 18px;
  padding: 22px;
  margin-bottom: 15px;
  border: 1px solid #edf2f7;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
 }}
 
 .chat-bubble-ai {{ background-color: #f1f5f9; padding: 12px; border-radius: 15px; margin-bottom: 10px; color: #334155; }}
 .chat-bubble-user {{ background-color: #22c55e; padding: 12px; border-radius: 15px; margin-bottom: 10px; color: white; text-align: right; }}

 /* 카드형 설문 답변 공통 디자인 */
 div[role="radiogroup"] {{
  gap: 10px !important;
 }}
 
 div[role="radiogroup"] > label {{
  background-color: white !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 12px !important;
  padding: 10px 15px !important;
  margin-bottom: 5px !important;
  width: 100% !important;
  display: flex !important;
  align-items: center !important;
  transition: all 0.2s ease !important;
 }}
 
 div[role="radiogroup"] > label:hover {{
  border-color: #22c55e !important;
 }}

 div[role="radiogroup"] > label[data-checked="true"] {{
  border-color: #ff4b4b !important;
  background-color: #fffafa !important;
 }}

 div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] p {{
  color: #ff4b4b !important;
  font-weight: 600 !important;
 }}

 /* --- 핵심 수정: 성별(Gender) 버튼만 수평으로 강제 --- */
 /* 첫 번째 컬럼(성함) 옆의 두 번째 컬럼(성별) 라디오 그룹만 타겟팅 */
 div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[role="radiogroup"] {{
  flex-direction: row !important;
  display: flex !important;
  flex-wrap: wrap !important;
 }}
 
 div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[role="radiogroup"] > label {{
  width: auto !important;
  flex: 1 !important;
  min-width: 80px !important;
 }}

 button[kind="primary"] {{
  background-color: #ff4b4b !important;
  border: none !important;
 }}
</style>
""", unsafe_allow_html=True)

# --- 핵심 함수: 설문 점수 계산 ---
def calculate_scores():
 phq_mapping = {"전혀 아니다": 0, "여러 날 동안": 1, "일주일 이상": 2, "거의 매일": 3, "모름, 무응답": 0}
 phq = 0
 for v in st.session_state.survey_answers['PHQ9'].values():
  for key, val in phq_mapping.items():
   if key in v:
    phq += val
    break
 
 gad_mapping = {"전혀 아니다": 0, "며칠 동안": 1, "7일 이상": 2, "거의 매일": 3}
 gad = 0
 for v in st.session_state.survey_answers['GAD7'].values():
  for key, val in gad_mapping.items():
   if key in v:
    gad += val
    break
 
 bp1_score = 1
 if st.session_state.survey_answers['BP1']:
  ans = list(st.session_state.survey_answers['BP1'].values())[0]
  bp1_score = int(ans.split(".")[0]) if "." in ans else 1
 
 eq_ans = []
 for v in st.session_state.survey_answers['EQ5D'].values():
  if v:
   num = int(v.split(".")[0]) if "." in v else 1
   eq_ans.append(num)
  else:
   eq_ans.append(1)
 
 while len(eq_ans) < 5:
  eq_ans.append(1)
 
 m2, m3 = (1, 0) if eq_ans[0]==2 else (0, 1) if eq_ans[0]==3 else (0, 0)
 sc2, sc3 = (1, 0) if eq_ans[1]==2 else (0, 1) if eq_ans[1]==3 else (0, 0)
 ua2, ua3 = (1, 0) if eq_ans[2]==2 else (0, 1) if eq_ans[2]==3 else (0, 0)
 pd2, pd3 = (1, 0) if eq_ans[3]==2 else (0, 1) if eq_ans[3]==3 else (0, 0)
 ad2, ad3 = (1, 0) if eq_ans[4]==2 else (0, 1) if eq_ans[4]==3 else (0, 0)
 n3 = 1 if 3 in eq_ans else 0
 
 eq5d = 1 - (0.05 + 0.096*m2 + 0.418*m3 + 0.046*sc2 + 0.209*sc3 +
     0.038*ua2 + 0.192*ua3 + 0.058*pd2 + 0.278*pd3 +
     0.062*ad2 + 0.19*ad3 + 0.05*n3)
 
 return phq, gad, bp1_score, eq5d

# --- 핵심 함수: AI 예측 ---
def get_predictions():
 u = st.session_state.user_data
 bmi = u['weight'] / ((u['height']/100)**2)
 phq, gad, bp1, eq5d = calculate_scores()
 
 full_data = {
  'age': u['age'],
  'sex': 1 if u['gender'] == "남성" else 2,
  'edu': {"초졸 이하": 1, "중졸": 2, "고졸": 3, "대졸 이상": 4}.get(u['edu'], 3),
  'marry': {"기혼": 1, "미혼": 2, "이혼/사별/기타": 3}.get(u['marry'], 1),
  'FH_HE': 1 if "고혈압" in u['family_history'] else 0,
  'FH_DB': 1 if "당뇨병" in u['family_history'] else 0,
  'FH_DY': 1 if "이상지혈증" in u['family_history'] else 0,
  'FH_HAA': 1 if "뇌졸중" in u['family_history'] else 0,
  'HE_BMI': bmi,
  'alcohol': 1 if u['alcohol'] == "예" else 0,
  'mh_PHQ_S': phq,
  'mh_GAD_S': gad,
  'BP1': bp1,
  'EQ5D': eq5d,
  'sleep_time_wy': u['sleep_time'],
  'incm': {"하": 1, "중하": 2, "중상": 3, "상": 4}.get(u['incm'], 4)
 }
 
 predictions = {}
 for disease_name, model_info in MODELS.items():
  input_row = [full_data.get(feature, 0) for feature in model_info['features']]
  input_df = pd.DataFrame([input_row], columns=model_info['features'])
  prob = model_info['pipeline'].predict_proba(input_df)[0, 1]
  
  predictions[disease_name] = {
   "prob": prob,
   "threshold": model_info['threshold']
  }
 return predictions

# --- STEP 1: 건강 정보 입력 ---
if st.session_state.step == 1:
 with st.container():
  st.markdown('<div class="card-content">', unsafe_allow_html=True)
  st.markdown('<h2 style="text-align:center; margin-bottom:30px;">🏥 케어메이트</h2>', unsafe_allow_html=True)
  
  # c1(성함), c2(성별) -> CSS에서 nth-child(2)를 통해 성별만 수평 배치
  c1, c2 = st.columns(2)
  with c1:
   name = st.text_input("성함", value=st.session_state.user_data["name"])
  with c2:
   gender = st.radio("성별", ["남성", "여성"],
       index=0 if st.session_state.user_data["gender"]=="남성" else 1,
       horizontal=True)
  
  c3, c4 = st.columns(2)
  with c3:
   edu = st.selectbox("교육 수준", ["초졸 이하", "중졸", "고졸", "대졸 이상"], index=3)
  with c4:
   marry = st.selectbox("결혼 여부", ["기혼", "미혼", "이혼/사별/기타"], index=0)
  
  st.divider()
  
  col_a, col_b, col_c = st.columns(3)
  with col_a:
   age = st.number_input("나이 (세)", min_value=1, max_value=120,
            value=st.session_state.user_data["age"])
  with col_b:
   height = st.number_input("키 (cm)", min_value=50, max_value=250,
             value=st.session_state.user_data["height"])
  with col_c:
   weight = st.number_input("몸무게 (kg)", min_value=20, max_value=200,
             value=st.session_state.user_data["weight"])
  
  col_d, col_e, col_f = st.columns(3)
  with col_d:
   incm = st.selectbox("소득 수준", ["하", "중하", "중상", "상"], index=3)
  with col_e:
   # 음주 여부는 col_e에 있어 수직 카드 형태를 유지함
   alcohol = st.radio("음주 여부", ["아니오", "예"], horizontal=False)
  with col_f:
   sleep = st.number_input("평균 수면시간 (시간)", min_value=0, max_value=24,
             value=st.session_state.user_data["sleep_time"])
  
  st.divider()
  
  diseases = st.multiselect("현재 진단받은 질환",
               ["고혈압", "당뇨병", "이상지혈증", "뇌졸중"],
               default=st.session_state.user_data["diseases"])
  
  family_history = st.multiselect("가족력 (부모/형제자매)",
               ["고혈압", "당뇨병", "이상지혈증", "뇌졸중"],
               default=st.session_state.user_data["family_history"])
  
  st.session_state.user_data.update({
   "name": name, "gender": gender, "age": age, "height": height,
   "weight": weight, "diseases": diseases, "family_history": family_history,
   "edu": edu, "marry": marry, "incm": incm, "alcohol": alcohol,
   "sleep_time": sleep
  })
  
  st.divider()
  st.write("### 📋 입력하신 정보가 정확합니까?")
  
  col1, col2 = st.columns(2)
  with col1:
   if st.button("✅ 네, 맞습니다", type="primary", use_container_width=True):
    if not name:
     st.error("성함을 입력해 주세요.")
    else:
     st.session_state.data_confirmed = True
     st.rerun()
  with col2:
   if st.button("🔄 수정하겠습니다", use_container_width=True):
    st.session_state.data_confirmed = False
    st.info("상단 입력란에서 내용을 수정해 주세요.")

  if st.session_state.data_confirmed:
   st.success("✅ 데이터가 저장되었습니다.")
   if st.button("다음 단계: 정신건강 설문 ➡", type="primary", use_container_width=True):
    st.session_state.step = 2
    st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 2: 정신건강 설문 ---
elif st.session_state.step == 2:
 SURVEY_DATA = {
  1: {
   "title": "📋 PHQ-9 (우울증 설문)",
   "questions": [
    "1. 일을 하는 것에 대한 흥미나 재미가 거의 없음",
    "2. 기분이 가라앉거나 우울하거나 희망이 없다고 느꼈다",
    "3. 잠들기 어렵거나 자주 깨거나 너무 많이 잤다",
    "4. 피곤하고 기력이 거의 없었다",
    "5. 식욕이 저하되거나 과식을 했다",
    "6. 자신이 실패자라고 느끼거나 자신 또는 가족을 실망시켰다",
    "7. 신문을 읽거나 TV를 보는 것과 같은 일에 집중하기 어려웠다",
    "8. 다른 사람들이 알아챌 정도로 너무 느리게 움직이거나 말을 했다",
    "9. 자신을 해치거나 차라리 죽는 것이 낫겠다는 생각을 했다"
   ],
   "options": ["전혀 아니다", "여러 날 동안", "일주일 이상", "거의 매일", "모름, 무응답"],
   "key": "PHQ9"
  },
  2: {
   "title": "😰 GAD-7 (불안도 설문)",
   "questions": [
    "1. 초조하거나 불안하거나 조마조마하게 느낀다",
    "2. 걱정하는 것을 멈추거나 조절할 수 없다",
    "3. 여러 가지 것들에 대해 걱정을 너무 많이 한다",
    "4. 편하게 있기가 어렵다",
    "5. 너무 안절부절못해서 가만히 있기 힘들다",
    "6. 쉽게 짜증이 나거나 쉽게 성을 낸다",
    "7. 마치 끔찍한 일이 일어날 것처럼 두렵게 느낀다"
   ],
   "options": ["전혀 아니다", "며칠 동안", "7일 이상", "거의 매일"],
   "key": "GAD7"
  },
  3: {
   "title": "😓 BP1 (스트레스 인지)",
   "questions": ["평소 일상생활 중 스트레스를 어느 정도 느끼십니까?"],
   "options": ["1. 거의 느끼지 않음", "2. 조금 느끼는 편이다", "3. 많이 느끼는 편이다", "4. 대단히 많이 느낀다"],
   "key": "BP1"
  },
  4: {
   "title": "💪 EQ5D (삶의 질)",
   "questions": ["4-1. 운동능력", "4-2. 자기관리", "4-3. 일상활동", "4-4. 통증/불편", "4-5. 불안/우울"],
   "options_per_question": [
    ["1. 걷는데 지장이 없음", "2. 걷는데 다소 지장이 있음", "3. 종일 누워 있어야 함"],
    ["1. 목욕이나 옷 입는데 지장 없음", "2. 목욕이나 옷 입는데 다소 지장 있음", "3. 혼자 목욕하거나 옷 입기 힘듦"],
    ["1. 일상 활동에 지장 없음", "2. 일상 활동에 다소 지장 있음", "3. 일상 활동을 할 수 없음"],
    ["1. 통증이나 불편감 없음", "2. 다소 통증이나 불편감 있음", "3. 매우 심한 통증이나 불편감 있음"],
    ["1. 불안하거나 우울하지 않음", "2. 다소 불안하거나 우울함", "3. 매우 불안하거나 우울함"]
   ],
   "key": "EQ5D"
  }
 }
 
 curr = SURVEY_DATA[st.session_state.sub_step]
 q_idx = st.session_state.q_idx
 
 with st.container():
  st.markdown('<div class="card-content">', unsafe_allow_html=True)
  st.markdown(f'<h3 style="color:#22c55e; margin-bottom:5px;">{curr["title"]}</h3>', unsafe_allow_html=True)
  
  
  total_q = len(curr['questions'])
  st.progress((q_idx + 1) / total_q)
  st.caption(f"문항 {q_idx + 1} / {total_q}")
  
  st.markdown(f"#### {curr['questions'][q_idx]}")
  
  opts = (curr["options_per_question"][q_idx] if "options_per_question" in curr else curr["options"])
  
  answer = st.radio("Select an answer", opts, key=f"q_{st.session_state.sub_step}_{q_idx}", label_visibility="collapsed")
  st.session_state.survey_answers[curr["key"]][f"q{q_idx}"] = answer
  
  st.markdown("<br>", unsafe_allow_html=True)
  
  b1, b2 = st.columns(2)
  with b1:
   if st.button("⬅ 이전 질문", use_container_width=True):
    if q_idx > 0:
     st.session_state.q_idx -= 1
    elif st.session_state.sub_step > 1:
     st.session_state.sub_step -= 1
     st.session_state.q_idx = len(SURVEY_DATA[st.session_state.sub_step]["questions"]) - 1
    else:
     st.session_state.step = 1
    st.rerun()
  
  with b2:
   if q_idx < len(curr["questions"]) - 1:
    button_text = "다음 질문 ➡"
   elif st.session_state.sub_step < 4:
    button_text = "다음 설문 ➡"
   else:
    button_text = "분석 결과 보기 🎯"
   
   if st.button(button_text, type="primary", use_container_width=True):
    if q_idx < len(curr["questions"]) - 1:
     st.session_state.q_idx += 1
    elif st.session_state.sub_step < 4:
     st.session_state.sub_step += 1
     st.session_state.q_idx = 0
    else:
     st.session_state.step = 3
    st.rerun()
  st.markdown('</div>', unsafe_allow_html=True)

# --- STEP 3: AI 분석 리포트 ---
elif st.session_state.step == 3:
 st.markdown("<h2 style='text-align:center; margin-bottom:30px;'>📊 AI 건강 분석 리포트</h2>", unsafe_allow_html=True)
 
 u = st.session_state.user_data
 bmi = u['weight'] / ((u['height']/100)**2)
 # 점수 계산 함수에서 bp1_score도 가져옵니다.
 phq, gad, bp1_score, eq5d = calculate_scores()
 
 # 요약 바에 스트레스 지수(bp1_score)를 추가했습니다.
 st.markdown(f"""
 <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px;">
  <h3 style="margin:0; color: white;">👤 {u['name']}님의 건강 프로필</h3>
  <p style="margin:5px 0; color: white;">나이: {u['age']}세 | 성별: {u['gender']} | BMI: {bmi:.1f}</p>
  <p style="margin:5px 0; color: white;">우울: {phq}점 | 불안: {gad}점 | 스트레스: {bp1_score}점 | 삶의 질: {eq5d:.2f}</p>
 </div>
 """, unsafe_allow_html=True)
 
 preds = get_predictions()
 high_risks, mid_risks = [], []
 risk_summary_text = []
 
 for d_name, res in preds.items():
  prob, threshold = res['prob'], res['threshold']
  score = int(prob * 100)
  
  if prob >= threshold: level = "높음"; high_risks.append(d_name)
  elif prob >= threshold * 0.7: level = "중간"; mid_risks.append(d_name)
  else: level = "낮음"
  
  if level in ["높음", "중간"]: risk_summary_text.append(f"{d_name}({level})")
  
  theme = LEVEL_THEMES[level]
  st.markdown(f"""
  <div class="disease-item-card">
   <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
    <span style="font-weight: bold; font-size: 1.2rem; color: #334155;">{theme['emoji']} {d_name}</span>
    <div style="text-align: right;">
     <span style="color: {theme['color']}; font-weight: bold; font-size: 1.2rem;">{level}</span>
     <span style="color: #64748b; font-size: 0.9rem; margin-left: 8px;">위험도 {score}점</span>
    </div>
   </div>
   <div style="width: 100%; background-color: #f1f5f9; border-radius: 10px; height: 14px; overflow: hidden;">
    <div style="width: {score}%; background-color: {theme['color']}; height: 100%; border-radius: 10px;"></div>
   </div>
   <p style="margin-top: 10px; color: #64748b; font-size: 0.9rem;">발병 확률: {prob:.1%} | 기준 임계값: {threshold:.1%}</p>
  </div>
  """, unsafe_allow_html=True)

 st.session_state.risks_summary = ", ".join(risk_summary_text) if risk_summary_text else "정상"

 st.write("---")
 st.markdown("### 💡 종합 의견")
 if high_risks: st.error(f"**고위험 질환**: {', '.join(high_risks)} - 전문의 상담을 권장합니다.")
 if mid_risks: st.warning(f"**중위험 질환**: {', '.join(mid_risks)} - 생활습관 개선이 필요합니다.")
 if not high_risks and not mid_risks: st.success("모든 질환이 저위험입니다. 현재 상태를 유지하세요!")
 
 st.write("---")
 c1, c2 = st.columns(2)
 with c1:
  if st.button("🎙️ AI 상담 시작하기", type="primary", use_container_width=True):
   st.session_state.chat_history = [{"role": "ai", "content": f"안녕하세요 {st.session_state.user_data['name']}님. 분석 결과 {st.session_state.risks_summary} 위험이 확인되었습니다. 어떤 점을 도와드릴까요?"}]
   st.session_state.step = 4; st.rerun()
 with c2:
  if st.button("🔄 처음으로 돌아가기", use_container_width=True):
   for key in list(st.session_state.keys()): del st.session_state[key]
   st.rerun()

# --- STEP 4: AI 음성 챗봇 상담 (Edge TTS 적용) ---
elif st.session_state.step == 4:
 
 # Edge TTS 음성 생성 함수
 async def generate_edge_tts_async(text):
  """Edge TTS로 음성 생성 (비동기)"""
  try:
   communicate = edge_tts.Communicate(text, "ko-KR-SunHiNeural")
   audio_data = b""
   
   async for chunk in communicate.stream():
    if chunk["type"] == "audio":
     audio_data += chunk["data"]
   
   return audio_data
  except Exception as e:
   return None
 
 def generate_edge_tts(text):
  """Edge TTS 동기 래퍼 함수"""
  try:
   # 이벤트 루프 생성 및 실행
   loop = asyncio.new_event_loop()
   asyncio.set_event_loop(loop)
   audio_data = loop.run_until_complete(generate_edge_tts_async(text))
   loop.close()
   return audio_data
  except Exception as e:
   return None
 
 with st.container():
  st.markdown('<div class="card-content">', unsafe_allow_html=True)
  st.subheader("🤖 AI 건강 비서")
  
  # 채팅 히스토리 표시 영역
  chat_container = st.container()
  with chat_container:
   for idx, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "user":
     st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
     st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
     if "audio" in msg: 
      st.audio(msg["audio"], format="audio/mp3")
  
  # 입력 영역
  col1, col2 = st.columns([4, 1])
  with col2:
   st.write("🎙️ 음성입력")
   voice_msg = speech_to_text(language='ko', just_once=True, key='stt_final')
  with col1:
   user_msg = st.chat_input("증상이나 궁금한 점을 물어보세요.")

  final_input = voice_msg if voice_msg else user_msg

  if final_input:
   # 사용자 메시지 추가
   st.session_state.chat_history.append({"role": "user", "content": final_input})
   
   # AI 응답 생성 영역
   with chat_container:
    st.markdown(f'<div class="chat-bubble-user">👤 {final_input}</div>', unsafe_allow_html=True)
    ai_message_placeholder = st.empty()
   
   try:
    # LLM 설정
    llm = ChatOpenAI(
     model="gpt-4o",
     api_key=OPENAI_API_KEY,
     temperature=0.7,
     streaming=True
    )
    
    u = st.session_state.user_data
    phq, gad, _, _ = calculate_scores()
    
    # 시스템 메시지
    sys_msg = (f"건강 상담사. 대상: {u['name']}({u['age']}세). "
               f"위험: {st.session_state.risks_summary}. "
               f"우울{phq}점, 불안{gad}점. 친절하고 구체적으로 답변.")
    
    # 스트리밍 응답
    full_response = ""
    
    for chunk in llm.stream([
     SystemMessage(content=sys_msg), 
     HumanMessage(content=final_input)
    ]):
     full_response += chunk.content
     ai_message_placeholder.markdown(
      f'<div class="chat-bubble-ai">🤖 {full_response}▌</div>', 
      unsafe_allow_html=True
     )
    
    # 최종 응답 표시
    ai_message_placeholder.markdown(
     f'<div class="chat-bubble-ai">🤖 {full_response}</div>', 
     unsafe_allow_html=True
    )
    
    # Edge TTS로 음성 생성
    audio_data = None
    with st.spinner("🔊 음성 생성 중..."):
     try:
      audio_data = generate_edge_tts(full_response)
      
      if audio_data:
       with chat_container:
        st.audio(audio_data, format="audio/mp3")
      else:
       st.warning("음성 생성에 실패했습니다.")
       
     except Exception as tts_error:
      st.warning(f"음성 생성 실패: {tts_error}")
    
    # 히스토리 저장
    chat_entry = {"role": "ai", "content": full_response}
    if audio_data:
     chat_entry["audio"] = audio_data
    
    st.session_state.chat_history.append(chat_entry)
    st.rerun()
    
   except Exception as e:
    st.error(f"상담 중 오류: {e}")

  if st.button("⬅ 결과 리포트로 돌아가기"): 
   st.session_state.step = 3
   st.rerun()
  
  st.markdown('</div>', unsafe_allow_html=True)