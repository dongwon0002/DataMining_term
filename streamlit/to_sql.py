import pandas as pd
import sqlite3
from sklearn.preprocessing import LabelEncoder

# CSV 읽기
df = pd.read_csv("./final_season_added.csv")
df = df.drop([
    '전월세구분','계약일','도로명','구','동','계약년','시작연','시작월','종료연','종료월','위도','경도','geometry','행정동','동별','건물명'
],axis=1)


le = LabelEncoder()
df['주택유형_encoded'] = le.fit_transform(df['주택유형'])
df = pd.get_dummies(df, columns=['계절'], prefix='계절', drop_first=True)
df = df.drop(['주택유형'],axis=1)
# SQLite DB 저장
conn = sqlite3.connect("data.db")
df.to_sql("my_table", conn, if_exists="replace", index=False)
conn.close()
