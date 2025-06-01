1. 전국 음식점 데이터 전처리
   - ```영업/정상```인경우만 남기고 제거
   - 불필요 컬럼 제거
   - 서울시 개방자치단체코드만 남기
```python
re = re[re['영업상태명']=='영업/정상']
re.drop(['번호','개방서비스아이디','관리번호','인허가일자','인허가취소일자','영업상태구분코드',
         '폐업일자','휴업시작일자','재개업일자','소재지전화','소재지우편번호','최종수정시점','데이터갱신구분',
         '데이터갱신일자','업태구분명','위생업태명','남성종사자수','여성종사자수','영업장주변구분명','등급구분명',
         '급수시설구분명','총직원수','본사직원수','공장사무직직원수','공장판매직직원수','공장생산직직원수','건물소유구분명',
         '보증액','월세액','다중이용업소여부','시설총규모','전통업소지정번호','전통업소주된음식','홈페이지','Unnamed: 47'],axis=1, inplace=True)
seoul_sigungu_codes = [
    3000000, 3010000, 3020000, 3030000, 3040000,
    3050000, 3060000, 3070000, 3080000, 3090000,
    3100000, 3110000, 3120000, 3130000, 3140000,
    3150000, 3160000, 3170000, 3180000, 3190000,
    3200000, 3210000, 3220000, 3230000, 3240000
]
# 해당 코드만 포함된 행만 필터링
df_filtered = re[re['개방자치단체코드'].isin(seoul_sigungu_codes)].copy()

# 결과 확인
print(df_filtered.info())
```
   - ```도로명전체주소```를 바탕으로 ```위도```, ```경도```컬럼을 생성 (kakao map API사용)
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


# 2. 주소별 위도/경도 저장 딕셔너리
address_to_latlng = {}

for addr in loc_list:
    lat, lng = get_lat_lng(addr)
    address_to_latlng[addr] = (lat, lng)
    time.sleep(0.2)  # API 과다 호출 방지

df_filtered['위도'] = df_filtered['도로명전체주소'].map(lambda x: address_to_latlng.get(x, (None, None))[0])
df_filtered['경도'] = df_filtered['도로명전체주소'].map(lambda x: address_to_latlng.get(x, (None, None))[1])
```
   - BallTree 알고리즘을 이용하여 거리기준 내 음식점 개수 관련 컬럼 생성
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
rest00_rad = to_radians(df_filtered, '위도', '경도')

# BallTree 생성 (Haversine metric 사용)
tree = BallTree(rest00_rad, metric='haversine')

# 매물별 거리 내 병원 개수 계산
for r, km in zip(radii, radius_km):
    count = tree.query_radius(df00_rad, r=r, count_only=True)
    df00[f'식당_{km}km내_개수'] = count
```

2. 서울시 지구대/파출소 데이터 전처리
   - 위경도 컬럼 추출 => 위와 동일 코드 생략
   - 카카오맵 API를 이용해도 나오지 않은 위경도는 확인후 직접 입력
```python
pol.loc[pol['관서명']=='염창','위도']=37.55525
pol.loc[pol['관서명']=='염창','경도']=126.8715
pol.loc[pol['관서명']=='도봉1','위도']=37.67945
pol.loc[pol['관서명']=='도봉1','경도']=127.0434
```
   - BallTree알고리즘을 이용해 거리기준 내 지구대/파출소 컬럼 생성 => 위와 동일 코드 생략
      
3. 공원데이터를 이용한 기준거리 이내 공원 개수 컬럼 생성    
   - [공원 전처리_readme.md](https://github.com/dongwon0002/DataMininig_term)

4. 지하철역 위치 데이터를 이용한 기준거리 이내 지하철 역 개수 컬럼 생성   
   - [지하철역 전처리_readme.md](https://github.com/dongwon0002/DataMininig_term)
 
5. 대학/대학원 데이터 전처리
   - 불필요 컬럼 제거
   - '대학명 단과대'형식의 이름을 분리
   - 소재지 서울만 유지
```python
col.drop(['학교 영문명','설립형태구분명', '소재지지번주소','도로명우편번호','소재지우편번호','홈페이지주소','대표팩스번호','설립일자','기준연도','데이터기준일자','제공기관코드''제공기관명'],axis=1,inplace=True)

col['uni_name'] = col['학교명'].str.split(expand=True)[0]

col = col[col['시도명']=='서울특별시']
col.head()
```
   - ```소재지도로명주소```로 위경도 컬럼 생성 => 코드 생략
   - BallTree를 이용한 기준거리 내 대학교 개수 컬럼생성 => 코드 생략
