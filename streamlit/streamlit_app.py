import streamlit as st
import sqlite3
import pandas as pd

# DB 연결 함수
def get_connection():
    return sqlite3.connect("data.db")

# 쿼리 실행 함수
def run_query(query, params=None):
    conn = get_connection()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# Streamlit UI
st.title("CSV 필터링 및 시각화")

# 사용할 컬럼 목록 동적 생성
conn = get_connection()
all_columns = pd.read_sql_query("PRAGMA table_info(my_table)", conn)['name'].tolist()
conn.close()

# 사용자에게 컬럼 선택받기
selected_column = st.selectbox("필터할 컬럼을 선택하세요", all_columns)

# 고유 값 불러오기
unique_vals = run_query(f"SELECT DISTINCT [{selected_column}] FROM my_table")[selected_column].tolist()
selected_value = st.selectbox(f"{selected_column} 값 선택", unique_vals)

# 필터 쿼리 실행
query = f"SELECT * FROM my_table WHERE [{selected_column}] = ?"
df_filtered = run_query(query, (selected_value,))

# 결과 보여주기
st.subheader("필터링 결과")
st.dataframe(df_filtered)

# 시각화 예시 (바 차트로 특정 수치 컬럼 시각화)
numeric_columns = df_filtered.select_dtypes(include='number').columns.tolist()
if numeric_columns:
    col_to_plot = st.selectbox("시각화할 수치 컬럼 선택", numeric_columns)
    st.bar_chart(df_filtered[col_to_plot])
else:
    st.warning("시각화할 수치형 컬럼이 없습니다.")
