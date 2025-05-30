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

===========
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
 - ```시군구```컬럼 ```시```,```군```,```구```로 나누고, ```보증금(만원)```,```월세금(만원)```컬럼의  컴마(,)제거하고 숫자형 컬럼으로 변경, ```계약년월```컬럼에서 ```계약년```, ```계약월```컬럼으로 세분화후 숫자형 변수로 ```계약기간```컬럼에서 ```시작연,시작월,종료연,종료월```컬럼 추출후 ```계약개월수```컬럼 생
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


#### 2. 불필요한 컬럼 제거

#### 3. 컬럼 추가(위경도, 거리기반 경찰 병원...등 어떻게 썻는지)
 (컬럼 추가 정리 README 연결)
 
#### 4. 타겟 변경- 월 부담액(왜 월세를 안쓰기로 했는지)
 이게 왜 현실적인지에 대한 설명을 해놔야 나중에 할말이있다

### XGBoost 모델링

#### 알고리즘 선정 이유
 **XGBoost(eXtreme Gradient Boosting)**: 결정 트리 기반의 앙상블 회귀/분류 모델이며 Gradient Boosting 방식을 빠르고 정밀하게 구현한 알고리즘이다. 
초기에는 Random Forest를 고려했으나 40만 개가 넘는 객체의 방대한 규모를 다루기에는 모델 학습 시간과 효율성을 고려할 필요가 있었다. 
Random Forest의 경우 병렬성이 낮고 대규모일수록 학습 시간이 증가한다는 점과 다수의 feature를 고려하였을 때, 사용이 힘들 것이라 판단하였다. 
따라서, 더 빠르고 최적화된 학습이 가능하면서도 복잡한 변수 간 상호작용을 잘 학습할 수 있는 XGBoost를 선택하게 되었다.
(잔차(오차)를 예측하는 회귀 트리를 반복적으로 학습해서 Loss 기준으로 트리를 업데이트하며, L1,L2기반 정규화 및 2차 미분 기반으로 Loss Function 최적화하는 방식)


#### 결과 분석(아파트/다세대 분리 + 거리기반 각 1개만 선정)
 아파트+다세대 R^2 = 0.8/ 아파트 다세대 분리가 가능할 것 같아서 어 꼬ㅒ크내?? 한번 해보자

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



