import streamlit as st
import pandas as pd
import sqlite3
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import matplotlib.font_manager as fm
import platform

if platform.system() == 'Windows':
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
elif platform.system() == 'Darwin':  # macOS
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
else:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 리눅스 예시

font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 초기 설정
st.set_page_config(page_title="월세 영향 요인 분석", layout="wide")
load_dotenv()
DB_PATH = os.getenv("DATABASE_PATH")

# DB에서 데이터 불러오기
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM my_table", conn)
    conn.close()
    return df

# 데이터 로딩
df = load_data()

# 사용자 입력 받기
st.sidebar.header("입력 조건")
house_types = ['아파트', '연립다세대']
selected_house_type = st.sidebar.selectbox("주택유형 선택", house_types)
regions = df['법정동'].unique().tolist()
selected_region = st.sidebar.selectbox("법정동 선택", regions)
target_rent = st.sidebar.slider("희망 월세금 (만원)", min_value=10, max_value=500, step=5)

# 지역, 주택유형 필터링
type_map = {'아파트': 0, '연립다세대': 1}
filtered_df = df[(df['법정동'] == selected_region) & (df['주택유형_encoded'] == type_map[selected_house_type])].reset_index(drop=True)
if filtered_df.empty:
    st.warning("해당 조건의 데이터가 없습니다.")
    st.stop()

# 모델 로드 함수 분기
@st.cache_resource
def load_model_by_type(house_type):
    if house_type == '아파트':
        return joblib.load("model_apt.xgb")
    else:
        return joblib.load("model_villa.xgb")

model, model_columns = load_model_by_type(selected_house_type)

# 피처/타겟 설정
X = filtered_df.drop(columns=['월부담액', '법정동'], errors='ignore')
y = filtered_df['월부담액']

# 인코딩 및 컬럼 정렬
X_enc = pd.get_dummies(X)
for col in model_columns:
    if col not in X_enc.columns:
        X_enc[col] = 0
X_enc = X_enc[model_columns]

if X_enc.empty:
    st.error("필터링된 데이터가 없거나 피처가 모델과 일치하지 않습니다.")
    st.stop()

# 예측 및 SHAP
preds = model.predict(X_enc)
explainer = shap.TreeExplainer(model)
shap_exp = explainer(X_enc)

# 결과 표시
st.subheader(f"법정동: {selected_region} / 주택유형: {selected_house_type}의 월세 영향 요인 분석")
st.write(f"해당 조건 내 월부담액 평균: {y.mean():.1f}만원, 사용자가 희망하는 월세금: {target_rent}만원")

# SHAP summary plot과 waterfall plot을 한눈에 보기 좋게 컬럼에 배치
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 SHAP Summary Plot (전체 영향도)")
    fig1, ax1 = plt.subplots(figsize=(7, 3))
    shap.summary_plot(shap_exp, X_enc, show=False)
    st.pyplot(fig1)

with col2:
    nearest_idx = (y - target_rent).abs().idxmin()
    st.markdown(f"#### 🧠 Waterfall Plot (희망 월세 근접한 사례 기준 설명)")
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지 재적용
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    shap.plots.waterfall(shap_exp[nearest_idx], max_display=10, show=False)
    st.pyplot(fig2)
