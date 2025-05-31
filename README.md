## 서울시 월세예측&결정요인 분석
![image](https://github.com/user-attachments/assets/0126b02e-6635-4df4-a5cc-69273b7ac8c6)


### 배경 및 필요성
<pre>

 전세사기 증가로 월세 수요는 증가하는 가운데 월세에 대한 판단 기준이 부족하다고 판단하여
 분석 및 예측을 진행
 <기사출처>
 https://n.news.naver.com/article/022/0004031475?sid=101
 </pre>
  

#### 폴더 설명
(폴더명(설명-링크(링크는 마지막에 다 모아서)-- 폴더1: 데이터폴더/ 폴더2: 전처리폴더(코드 모음집느낌 왜냐? 전처리때 설명할꺼다~/ 폴더3: 분석 코드)

#### 파일 구성
data파일 : 사용된 데이터 모음 
L원본 데이터: 아파트(전월세)_실거래가_20250522185247.csv, 연립다세대(전월세)_실거래가_20250522185244.csv
|Prepocessing.ipynb  \>>> 매물 정보 + 추가 피쳐 생성을 위한 여러 데이터 추가

### 모델링
------------
#### 데이터 전처리 과정
원본데이터 전처리
```python
원본데이터 컬럼 >>>
Index(['시군구', '번지', '본번', '부번', '전월세구분', '전용면적(㎡)', '계약년월', '계약일',
       '보증금(만원)', '월세금(만원)', '층', '건축년도', '도로명', '계약기간', '계약구분',
       '갱신요구권 사용', '주택유형', '건물명'],
      dtype='object')
```

1.아파트, 연립다세대 데이터 통합
  - 같은 컬럼을 가리고 있어 두 데이터를 합친다
```python
combined = pd.concat([df_apt, df_vil], ignore_index=True)
```
2. 데이터 타입변환, 세부컬럼 생성 
  - ```시군구```컬럼 ```시```,```군```,```구```로 나눔
  - ```보증금(만원)```,```월세금(만원)```컬럼의  컴마(,)제거하고 숫자형 컬럼으로 변경
  - ```계약년월```컬럼에서 ```계약년```, ```계약월```컬럼으로 세분화후 숫자형 변수로
  - ```계약기간```컬럼에서 ```시작연,시작월,종료연,종료월```컬럼 추출후 ```계약개월수```컬럼 생성
```python
# 1. '시군구' 분리
com['시'] = com['시군구'].str.split(expand=True)[0]
com['구'] = com['시군구'].str.split(expand=True)[1]
com['동'] = com['시군구'].str.split(expand=True)[2]

# 2. 보증금, 월세금에서 ',' 제거 후 int로 변환
com['보증금(만원)'] = com['보증금(만원)'].astype(str).str.replace(',', '', regex=False).astype(int)
com['월세금(만원)'] = com['월세금(만원)'].astype(str).str.replace(',', '', regex=False).astype(int)

# 3. 계약년월 → 계약년, 계약월 분리
com['계약년'] = com['계약년월'].astype(str).str[:4].astype(int)
com['계약월'] = com['계약년월'].astype(str).str[4:].astype(int)

# 계약기간을 시작과 종료로 나누기
com[['계약시작', '계약종료']] = com['계약기간'].str.split('~', expand=True)

# 연도와 월로 분리
com['시작연'] = com['계약시작'].str[:4].astype(int)
com['시작월'] = com['계약시작'].str[4:].astype(int)
com['종료연'] = com['계약종료'].str[:4].astype(int)
com['종료월'] = com['계약종료'].str[4:].astype(int)

# 개월 수 계산
com['계약개월수'] = (com['종료연'] - com['시작연']) * 12 + (com['종료월'] - com['시작월'])

# 사용한 컬럼 제거하고 계약기간을 계약개월수로 대체하려면:
# com.drop(columns=['계약기간', '계약시작', '계약종료', '시작연', '시작월', '종료연', '종료월'], inplace=True)
```
3. 불필요 컬럼 제거
```python
com.drop(['계약구분','갱신요구권 사용','시군구','계약년월','번지','본번','부번'],axis=1, inplace=True)
```
4. 모델 타겟 후보 피쳐등 생성
 - 테스트를 위한 타겟 컬럼을 여럿 생성    
 자기자본으로 보증금을 마련하는 경우 (기회비용)
$$[
\text{월 부담액} = \left( \frac{\text{보증금} \times \text{연 이자율}}{12} \right) + \text{월세}
]$$    
대출로 보증금을 마련하는 경우 (대출이자만) => 원금상환은 제외(전세금 대출 유형)
$$[
\text{월 부담액} = \left( \frac{\text{보증금} \times 0.7 \times \text{연 대출금리}}{12} \right) + \text{월세}
]$$
```python
com['보증금/월세금'] = com['보증금(만원)']/com['월세금(만원)']
com['보증금/월세금'].round(2)

com['면적/월세금'] = com['전용면적(㎡)']/com['월세금(만원)']
com['면적/월세금'] = com['면적/월세금'].round(2)

com['월부담액'] = com['보증금(만원)']*0.04/12+com['월세금(만원)']
com['월부담액'] = com['월부담액'].round(2)

com['월세금/면적'] = com['월세금(만원)']/df00['전용면적(㎡)']
com['월세금/면적'] = com['면적/월세금'].round(2)
```
5. ```도로명``` 컬럼을 이용해 ```위도```, ```경도```컬럼 생성
  -  카카오맵 API를 사용하여 위경도 추출
```python
import requests
import pandas as pd
import time

def get_lat_lng(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": address}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        if result['documents']:
            location = result['documents'][0]['address']
            return float(location['y']), float(location['x'])  # 위도, 경도
    return None, None

# 1. 고유한 도로명 주소 리스트 추출
unique_addresses = com['도로명'].dropna().unique()

# 2. 주소별 위도/경도 저장 딕셔너리
address_to_latlng = {}

for addr in unique_addresses:
    lat, lng = get_lat_lng(addr)
    address_to_latlng[addr] = (lat, lng)
    time.sleep(0.2)  # API 과다 호출 방지

# 3. 원래 데이터프레임에 위도/경도 추가
com['위도'] = com['도로명'].map(lambda x: address_to_latlng.get(x, (None, None))[0])
com['경도'] = com['도로명'].map(lambda x: address_to_latlng.get(x, (None, None))[1])
````
6. ```위도```, ```경도```컬럼을 이용해 법정동, 행정도 데이터 생성
  - 법정동, 행정동 폴리곤 .shp파일을 이용하여 각각의 매물이 어느 동에 위치하는지를 나타내는 컬럼생성
  - 이후 분석과 추가 데이터 통합에 사용
  - 통합할 추가 데이터와의 동 이름 통일
```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# # 1. 행정동 GeoDataFrame 불러오기
F1 = '/content/drive/MyDrive/DataMining/term project/data/BND_ADM_DONG_PG.shp'
gdf = gpd.read_file(F1, encoding='cp949')
# 2. EPSG:5179 좌표계로 가정하고 명시적으로 설정 (중부원점 기준)
# gdf.set_crs(epsg=5179, inplace=True)

# 3. 위경도 좌표계 (WGS84: EPSG:4326)로 변환
gdf = gdf.to_crs(epsg=4326)

# 2. com: 위경도 기반 매물 데이터 (예: '경도', '위도' 컬럼 있음)
# 매물 포인트로 변환
df00['geometry'] = com.apply(lambda row: Point(row['경도'], row['위도']), axis=1)
gdf_points = gpd.GeoDataFrame(com, geometry='geometry', crs='EPSG:4326')  # 위경도는 EPSG:4326

# 3. 좌표계 맞추기 (행정동 데이터에 맞추어 변환)
# gdf_points = gdf_points.to_crs(epsg=5181)

# 4. 공간 조인: 어떤 동의 폴리곤에 포함되는지 계산
joined = gpd.sjoin(gdf_points, gdf[['ADM_NM', 'geometry']], how='left', predicate='within')

# 5. 결과: joined['ADM_NM'] 이 해당 매물의 행정동 이름
# 원래 com에 컬럼 추가
com['행정동'] = joined['ADM_NM']

# 추가할 데이터와 다른 동 이름 통일
com.loc[com['행정동']=='금호2·3가동','행정동']='금호2.3가동'
com.loc[com['행정동']=='면목3·8동','행정동']='면목3.8동'
com.loc[com['행정동']=='상계3·4동','행정동']='상계3.4동'
com.loc[com['행정동']=='상계6·7동','행정동']='상계6.7동'
com.loc[com['행정동']=='종로1·2·3·4가동','행정동']='종로1.2.3.4가동'
com.loc[com['행정동']=='종로5·6가동','행정동']='종로5.6가동'
com.loc[com['행정동']=='중계2·3동','행정동']='중계2.3동'
```
7. 서울시 병원위치 정보 데이터를 이용해 기준거리 이내 병원개수 컬럼 생성
  - (0.2, 0.5, 1)km 거리 이내의 병원개수 컬럼을 추가
  - BallTree알고리즘에서 Haversine metric을 이용하여 계산
```python
import numpy as np
from sklearn.neighbors import BallTree

# 위경도 → 라디안 변환 (Haversine 거리 계산을 위해)
def to_radians(df, lat_col='위도', lon_col='경도'):
    return np.radians(df[[lat_col, lon_col]].values)

# 1km, 3km, 10km를 라디안으로 변환 (지구 반지름 약 6371km 기준)
radius_km = [0.2, 0.5, 1]
radii = [r / 6371.0 for r in radius_km]

# 매물 위치, 병원 위치를 라디안으로 변환
df00_rad = to_radians(df00, '위도', '경도')
hos00_rad = to_radians(hos_loc, '병원위도', '병원경도')

# BallTree 생성 (Haversine metric 사용)
tree = BallTree(hos00_rad, metric='haversine')

# 매물별 거리 내 병원 개수 계산
for r, km in zip(radii, radius_km):
    count = tree.query_radius(df00_rad, r=r, count_only=True)
    df00[f'병원_{km}km내_개수'] = count
```
(법정동추가 방법은 유사 내용 생략)

8. 전국 음식점 정보 데이터를 이용해 기준거리 이내 음식점개수 컬럼 생성
  - [음식점 전처리_readme.md](https://github.com/dongwon0002/DataMininig_term)
  - 추가 방법은 병원과 동일
9. 서울시 지구대/파출소 데이터를 이용해 기준거리 이내 지구대/파출소 개수 컬럼 생성
  - [지구대/파출소_readme.md](https://github.com/dongwon0002/DataMininig_term)
10. 공원데이터를 이용한 기준거리 이내 공원 개수 컬럼 생성
  - [공원 전처리_readme.md](https://github.com/dongwon0002/DataMininig_term)
11. 지하철역 위치 데이터를 이용한 기준거리 이내 지하철 역 개수 컬럼 생성
  - [지하철역 전처리_readme.md](https://github.com/dongwon0002/DataMininig_term)
12. 대학/대학원 데이터를 이용한 기준거리 이내 공원 개수 컬럼 생성
  - [대학/대학원 전처리_readme.md](https://github.com/dongwon0002/DataMininig_term)
13. 서울시 등록인구 통계 데이터와 원본데이터를 행정동을 기준으로 통합
```python
# 1. 필요한 컬럼만 추출
df_pop = pop[['동별', '연령별', '항목', '2024. 4/4']].copy()

# 2. 새로운 컬럼명 생성: "연령별_항목" 형태
df_pop['col_name'] = df_pop['연령별'] + '_' + df_pop['항목']

# 3. 피벗: 동별을 인덱스로, 새로운 컬럼명(col_name)을 컬럼으로
df_pivot = df_pop.pivot_table(index='동별', columns='col_name', values='2024. 4/4')

# 4. 컬럼 이름 초기화
df_pivot.reset_index(inplace=True)

# 5. df01에 병합 (동별 기준)
com = com.merge(df_pivot, left_on='행정동', right_on='동별', how='left')
```
14. 통합된 등록인구 데이터 이용해 ```합계_등록외국인```,```65세이상_인구비율```,```0-19대인구```,```20-34대인구```,```35-64대인구비```컬럼을 생성
```python
# 2. 65세 이상 인구 총합 계산 (계 컬럼 기준)
age_65_plus_cols = [
    '65~69세_계', '70~74세_계', '75~79세_계', '80~84세_계', '85~89세_계',
    '90~94세_계', '95~99세_계', '100세 이상_계'
]
foreigner_col = ['합계_등록외국인']
com['외국인_비율']= com[foreigner_col].sum(axis=1)/com['합계_계']
com['65세이상_인구비율'] = com[age_65_plus_cols].sum(axis=1)/com['합계_계']

# 연령대별 합 계산
com['0-19대인구'] = com[['0~4세_계', '5~9세_계', '10~14세_계', '15~19세_계']].sum(axis=1)
com['20-34대인구'] = com[['20~24세_계', '25~29세_계', '30~34세_계']].sum(axis=1)
com['35-64대인구'] = com[['35~39세_계', '40~44세_계', '45~49세_계', '50~54세_계', '55~59세_계', '60~64세_계']].sum(axis=1)

# 비율 계산
com['0-19대인구비'] = com['0-19대인구'] / com['합계_계']
com['20-34대인구비'] = com['20-34대인구'] / com['합계_계']
com['35-64대인구비'] = com['35-64대인구'] / com['합계_계']

# 필요 없으면 중간 컬럼 삭제
com.drop(columns=['0-19대인구', '20-34대인구', '35-64대인구'], inplace=True)
```
15. ```주택유형```컬럼으로 encoding
  - 0: 아파트, 1: 연립다세대의 값을 가지는 ```주택유형_encoded```컬럼생성
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
com['주택유형_encoded'] = com.fit_transform(df['주택유형'])
com = com.drop(['주택유형'],axis=1)
```
------------ 

#### 2. 불필요한 컬럼 제거

#### 3. 컬럼 추가(위경도, 거리기반 경찰 병원...등 어떻게 썻는지)
 (컬럼 추가 정리 README 연결)
 
#### 4. 타겟 변경- 월 부담액(왜 월세를 안쓰기로 했는지)
 이게 왜 현실적인지에 대한 설명을 해놔야 나중에 할말이있다
![Image](https://github.com/user-attachments/assets/c171479b-7288-45ab-9e5c-24c0135caab0)

### XGBoost 모델링

#### 알고리즘 선정 이유
 **XGBoost(eXtreme Gradient Boosting)**: 결정 트리 기반의 앙상블 회귀/분류 모델이며 Gradient Boosting 방식을 빠르고 정밀하게 구현한 알고리즘이다. 
초기에는 Random Forest를 고려했으나 40만 개가 넘는 객체의 방대한 규모를 다루기에는 모델 학습 시간과 효율성을 고려할 필요가 있었다. 
Random Forest의 경우 병렬성이 낮고 대규모일수록 학습 시간이 증가한다는 점과 다수의 feature를 고려하였을 때, 사용이 힘들 것이라 판단하였다. 
따라서, 더 빠르고 최적화된 학습이 가능하면서도 복잡한 변수 간 상호작용을 잘 학습할 수 있는 XGBoost를 선택하게 되었다.
(잔차(오차)를 예측하는 회귀 트리를 반복적으로 학습해서 Loss 기준으로 트리를 업데이트하며, L1,L2기반 정규화 및 2차 미분 기반으로 Loss Function 최적화하는 방식)


#### 결과 분석(아파트/다세대 분리 + 거리기반 각 1개만 선정)
 아파트+다세대 R^2 = 0.8/ 아파트 다세대 분리가 가능할 것 같아서 어 꼬ㅒ크내?? 한번 해보자

##타켓 변수 선택

```python
#스케일링할 타겟 후보 컬럼 리스트
target_cols = ['보증금(만원)', '월세금(만원)', '월부담액', '보증금/월세금', '월세금/면적']

#1. 스케일러 선택 (MinMaxScaler or StandardScaler)
scaler = StandardScaler()

#2. 해당 컬럼들만 스케일링
scaled_values = scaler.fit_transform(df[target_cols])

# 3. 스케일된 결과를 데이터프레임으로 변환 (컬럼명 유지)
df_scaled = pd.DataFrame(scaled_values, columns=target_cols)

# 4. 원본 데이터에서 해당 컬럼들 제거 후 스케일된 컬럼으로 대체
df.drop(columns=target_cols, inplace=True)
df = pd.concat([df, df_scaled], axis=1)
```


## ```train_eval_xgb``` 함수가 하는 것
데이터를 train_test_split으로 나누고 train 셋에 대해 XGBRegressor모델을 생성하고, test 셋에 대해 RMSE, MAE, R^2값을 반환하고, 모델의 feature importance 그래프와 shap 그래프를 반환 가능하고 그래프는 True, False로 선택가능하다
### 파라미터
- ```df```
- ``` target_col```
- ```test_size```
- ```random_state```
- ``` plot_feature_importance```(bool)
- ```plot_shap```(bool)
### return
- ```model, (X_train, X_test, y_train, y_test)```      
---------

(ex)
```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df.drop(['월부담액','보증금/월세금','월세금/면적','보증금(만원)'],axis=1),'월세금(만원)', plot_feature_importance=False, plot_shap=False)
```
\>>> df.drop(['월부담액','보증금/월세금','월세금/면적','보증금(만원)'],axis=1)데이터를 바탕으로 모델을 구성하고 ```'월세금(만원)'```을 타겟으로 선정, 두개 그래프 모두 그리지 않는다


#### 모델 개선(아파트/다세대 분리> 각각 평가 > Feature importance로 거리기반 각각 중요만남긴다)

#### 최종 모델(아파트/다세대) 
 >>다시 feature importance 어떤 요인이 target에 영향?

### 시각화 (서울구별로 평귤 월부담액(아파트/다세대) 노원구만 법정동 따서 (아파트/다세대))

####  SHAP
 법정동 노원구 내에서 공릉1동 2동 하계 동 평균 featruer 대입해서 
 이 동에서는 평균 월 부담액이 70만원이고
 어떤 요인이 12만원 어던 요인이 5만원~ >>차트로 
 (모델을 통한 새로운 데이터 fitting시 결과 값의 기여도를 알 수 있다.)


### 시사점



