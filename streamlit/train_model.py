import pandas as pd
import sqlite3
import xgboost as xgb
import joblib
from dotenv import load_dotenv
import os

# 환경 변수 로딩
load_dotenv()
DB_PATH = os.getenv("DATABASE_PATH")

# 데이터 불러오기
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM my_table", conn)
conn.close()

# 불필요한 컬럼 제거
drop_cols = ['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 결측치 제거
df = df.dropna()

# 주택유형별 피처 리스트 정의
apt_features = [
    '전용면적(㎡)','자치구코드','건축년도','층','법정동코드','아파트_거래수',
    '계약개월수','계약월','35-64대인구비','연립다세대_거래수','0-19대인구비','20-34대인구비','65세이상_인구비율','외국인_비율',
    '병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수'
]
villa_features = [
    '병원_1km내_개수', '1km_이내_역_개수', '식당_0.5km내_개수', '공원_500m_이내_개수', '파출소/지구대_0.5km내_개수', '대학_1km내_개수',
    '전용면적(㎡)','자치구코드','건축년도','층','법정동코드','아파트_거래수',
    '계약개월수','계약월','35-64대인구비','연립다세대_거래수','0-19대인구비','20-34대인구비','65세이상_인구비율','외국인_비율'
]

# 주택유형별 데이터 분리
apt_df = df[df['주택유형_encoded'] == 0]
villa_df = df[df['주택유형_encoded'] == 1]

# 아파트 모델 학습 및 저장
if not apt_df.empty:
    X_apt = apt_df[[col for col in apt_features if col in apt_df.columns]]
    y_apt = apt_df['월부담액']
    X_apt = pd.get_dummies(X_apt)
    model_apt = xgb.XGBRegressor()
    model_apt.fit(X_apt, y_apt)
    joblib.dump((model_apt, X_apt.columns.tolist()), "model_apt.xgb")

# 연립다세대 모델 학습 및 저장
if not villa_df.empty:
    X_villa = villa_df[[col for col in villa_features if col in villa_df.columns]]
    y_villa = villa_df['월부담액']
    X_villa = pd.get_dummies(X_villa)
    model_villa = xgb.XGBRegressor()
    model_villa.fit(X_villa, y_villa)
    joblib.dump((model_villa, X_villa.columns.tolist()), "model_villa.xgb")
