# MySQL Workbench, Python 활용 분석

![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/c8787384-1090-4552-882c-887669d788be)


# MySQL Workbench 데이터 분석과 파이썬을 이용한 시각화

데이터 가져오기

```sql
SELECT * FROM titanic.train;
SELECT Sex, AVG(Survived) as survival_rate
FROM train
GROUP BY Sex;
```

![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/7837c372-dece-4a00-981a-370a86c5005a)

데이터 컬럼은 다음과 같습니다

PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/1e6db166-6c12-41f1-b867-8e2e88c3a49e)



# 데이터 전처리

일단 이름에 'Miss.', 'Mrs.', 'Mr.', 'Ms.', 'Null.' 값에 대해 분류하고 평균 연령을 계산합니다

### 각 타이틀에 대한 평균 연령과 생존율, 비율을 계산합니다.

```sql
SELECT 
  CASE 
    WHEN Name LIKE '%Miss.%' THEN 'Miss.'
    WHEN Name LIKE '%Mrs.%' THEN 'Mrs.'
    WHEN Name LIKE '%Mr.%' THEN 'Mr.'
    WHEN Name LIKE '%Ms.%' THEN 'Ms.'
    ELSE 'Others'
  END AS Title,
  AVG(Age) AS AverageAge,
  SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) / COUNT(Survived) * 100 AS SurvivalRate,
  COUNT(*) / (SELECT COUNT(*) FROM train WHERE Age IS NOT NULL) * 100 AS TitleRate
FROM train
WHERE Age IS NOT NULL
GROUP BY Title;
```
| Title  | AverageAge | SurvivalRate | TitleRage |
|--------|----------|--------------|-----------|
|  Mrs.  |  35.8981  |    78.7037   |  15.1261  |
| Miss.  |  21.7877  |    71.9178   |  20.4482  |
|  Mr.   |  32.3869  |    16.8342   |  55.7423  |
|  Ms.   |  28.0000  |    100.0000  |  0.1401   |
| Others |  20.3115  |    52.4590   |  8.5434   |

- Ms(여성) > Mrs(기혼여성) > Miss(미혼여성) > Others > Mr(남성) 순으로 생존율이 높습니다
- 여기에 연령이 표시되지 않은 other에 평균연령으로 계산.
- 
```sql
UPDATE train
SET Age = (SELECT AVG(Age) FROM train WHERE Name LIKE '%Miss.%' AND Age IS NOT NULL)
WHERE Age IS NULL AND Name LIKE '%Miss.%';

UPDATE train
SET Age = (SELECT AVG(Age) FROM train WHERE Name LIKE '%Mrs.%' AND Age IS NOT NULL)
WHERE Age IS NULL AND Name LIKE '%Mrs.%';

UPDATE train
SET Age = (SELECT AVG(Age) FROM train WHERE Name LIKE '%Mr.%' AND Age IS NOT NULL)
WHERE Age IS NULL AND Name LIKE '%Mr.%';

UPDATE train
SET Age = (SELECT AVG(Age) FROM train WHERE Name LIKE '%Ms.%' AND Age IS NOT NULL)
WHERE Age IS NULL AND Name LIKE '%Ms.%';
```

```sql
SELECT 
  CASE 
    WHEN Name LIKE '%Miss.%' THEN 'Miss.'
    WHEN Name LIKE '%Mrs.%' THEN 'Mrs.'
    WHEN Name LIKE '%Mr.%' THEN 'Mr.'
    WHEN Name LIKE '%Ms.%' THEN 'Ms.'
    ELSE 'Others'
  END AS Title,
  AVG(Age) AS AverageAge,
  SUM(CASE WHEN Survived = 1 THEN 1 ELSE 0 END) / COUNT(Survived) * 100 AS SurvivalRate,
  COUNT(*) / (SELECT COUNT(*) FROM train) * 100 AS TitleRate
FROM train
GROUP BY Title;
```

|  Title | AverageAge | SurvivalRate | TitleRage |
|--------|----------|--------------|-----------|
|  Mrs.  |  35.8981  |    78.7037   |  15.1261  |
| Miss.  |  21.7877  |    71.9178   |  20.4482  |
| Others |  20.3115  |    52.4590   |  8.5434   |
|  Mr.   |  32.3869  |    16.8342   |  55.7423  |
|  Ms.   |  28.0000  |    100.0000  |  0.1401   |

- 계산 결과는 다음과 같고. 데이터를 분석해본 결과 절반 이상의 otheres는 어린 남자 였습니다.
- 기혼여성 > 미혼여성 > 어린남성 > 어른남성 순으로 생존율이 높은 것을 확인했습니다.






### 성별별 생존률 분석: 

- gender_submission 테이블과 passengers 테이블을 조인하여 성별별 생존율을 계산

```sql
SELECT sex, AVG(survived) as survival_rate
FROM passengers
JOIN gender_submission ON passengers.PassengerId = gender_submission.PassengerId
GROUP BY sex;
```

| Sex    | SurvivalRate |
|--------|----------|
|  male  |  0.2053  |
| female |  0.7548  |


- 결과는 예쌍대로 여자 생존율은 74%, 남자 생존율 19% 즉, 여자 탑승객 100명 중 74% 생존 남자 탑승객 100명 중 19% 생존율을 보여준다.


### 객실 등급별 생존률 분석: 

passengers 테이블을 사용하여 객실 등급별 생존률을 계산

```sql
SELECT Pclass, AVG(survived) as survival_rate
FROM passengers
GROUP BY Pclass;
```

| Pclass | survival_rate |
|--------|---------------|
|   1    |   0.6559      |
|   2    |   0.4798      |
|   3    |   0.2394      |
|--------|---------------|


- 분석 결과 3등석 생존율 24%, 2등석 47%, 1등석 63%로 나왔습니다
- 1등석 > 2등석 > 3등석 순으로 생존율이 높았고, 특히 3등석 생존율이 낮았습니다.


### 연령대별 생존률 분석: 연령대별로 분류하여 생존률을 계산

```sql
SELECT 
    CASE 
        WHEN Age < 10 THEN '0-9'
        WHEN Age < 20 THEN '10-19'
        WHEN Age < 30 THEN '20-29'
        WHEN Age < 40 THEN '30-39'
        WHEN Age < 50 THEN '40-49'
        WHEN Age < 60 THEN '50-59'
        WHEN Age < 70 THEN '60-69'
        WHEN Age < 80 THEN '70-79'
    END AS age_range,
    AVG(Survived) as survival_rate
FROM train
GROUP BY age_range;
```

| age_range | survival_rate |
|-----------|---------------|
|    0-9    |    0.6129     |
|   10-19   |    0.4020     |
|   20-29   |    0.3500     |
|   30-39   |    0.4371     |
|   40-49   |    0.3820     |
|   50-59   |    0.4167     |
|   60-69   |    0.3158     |
|   70-79   |    0.0000     |
|   others  |    1.0000     |
|-----------|---------------|

연령 별 생존율은 0-9: 61%, 10-19: 40%, 20-29 : 35%, 30-39 43%, 40-49 : 38%, 50-59 : 41%, 60-69 : 31%, 70-79 : 0%, 80- : 100%

다음과 같이 나왔는데요 나이가 매우 어린 0-9 의 생존율은 61%로 나오고 10-69 사이 31% ~ 43% 생존율을 보여줍니다. 

어린아이를 우선적으로 구조되었던 것을 알 수 있었습니다.



### 성별, 형제 및 자매가 있는 경우 생존율

```sql
SELECT
  Sex,
  SibSp,
  AVG(Survived) AS SurvivalRate
FROM
  train
GROUP BY
  Sex,
  SibSp;
```

결과는 다음과 같습니다.

| Sex    | SibSp | SurvivalRate |
|--------|-------|--------------|
| female | 0     | 0.7986       |
| female | 1     | 0.7444       |
| female | 2     | 0.7500       |
| female | 3     | 0.5000       |
| female | 4     | 0.3333       |
| female | 5     | 0.0000       |
| male   | 0     | 0.1835       |
| male   | 1     | 0.3226       |
| male   | 2     | 0.1538       |
| male   | 3     | 0.0000       |
| male   | 4     | 0.0833       |
| male   | 5     | 0.0000       |

표에서 알 수 있듯이, 

성별이 여자인 경우 형제 및 자매가 0~2명까지 생존율이 74% ~ 79%에 육박하는데 3명 이상부터는 50% 이하로 줄어드는 것을 확인 할 수 있습니다.

성별이 남자인 1명일 경우 32%로 제일 높았고 나머지는 18%이하로 낮은 생존율을 보였습니다.

즉, 성별이 여성이고 형제 자매가 없거나 1명, 2명 있을 경우 생존율이 높았습니다.


이를 시각화

```python
import pymysql
import numpy as np
import matplotlib.pyplot as plt

# 데이터베이스 연결
connection = pymysql.connect(host='localhost', user='root', password='zlfhrld84!', db='titanic')

try:
    with connection.cursor() as cursor:
        sql = "SELECT  Sex,  SibSp,  AVG(Survived) AS SurvivalRate FROM train GROUP BY  Sex,  SibSp;"
        cursor.execute(sql)
        result = cursor.fetchall()

        male_data = []
        female_data = []
        for row in result:
            if row[0] == 'male':  # 'Sex' 필드의 인덱스가 0
                male_data.append(row[2])  # 'SurvivalRate' 필드의 인덱스가 2
            else:
                female_data.append(row[2])

finally:
    connection.close()


bar_width = 0.35
index = np.arange(len(male_data))

plt.bar(index, male_data, bar_width, label='Male')
plt.bar(index + bar_width, female_data, bar_width, label='Female')

plt.title('Survival Rate by Gender and Number of Siblings/Spouses')
plt.xlabel('Number of Siblings/Spouses')
plt.ylabel('Survival Rate')
plt.xticks(index + bar_width / 2, range(len(male_data)))
plt.legend()
plt.show()

```

![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/186b12d4-36d1-4e17-8722-4451d0b32935)




```python
import pymysql
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터베이스 연결
connection = pymysql.connect(host='localhost', user='root', password='zlfhrld84!', db='titanic')

try:
    # SQL 쿼리 실행하여 결과를 DataFrame으로 가져오기
    with connection.cursor() as cursor:
        sql = "SELECT Sex, SibSp, AVG(Survived) AS SurvivalRate FROM train GROUP BY Sex, SibSp;"
        cursor.execute(sql)
        result = cursor.fetchall()
        df = pd.DataFrame(result, columns=['Sex', 'SibSp', 'SurvivalRate'])

finally:
    connection.close()

# 피벗 테이블로 데이터 변환
pivot_table = df.pivot_table(index='Sex', columns='SibSp', values='SurvivalRate', fill_value=0)

# 히트맵 그리기
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='YlGnBu', cbar_kws={'label': 'Survival Rate'})
plt.title('Survival Rate by Sex and Number of Siblings/Spouses')
plt.xlabel('Number of Siblings/Spouses')
plt.ylabel('Sex')
plt.show()

```

![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/dfcae807-4762-409a-b173-ff4e08e437cf)


'Parch' 칼럼은 타이타닉 호의 승객이 함께 탑승한 부모나 자녀의 수를 나타낸다.

과연 부모와 자녀 수를 통해 생존율에 어떠한 영향으 미치는지 알아보자

```sql
SELECT
  Parch,
  AVG(Survived) AS SurvivalRate
FROM
  train
GROUP BY
  Parch;
```

| SibSp | SurvivalRate |
|-------|--------------|
| 0     | 0.3570       |
| 1     | 0.5545       |
| 2     | 0.5735       |
| 3     | 0.6000       |
| 4     | 0.0000       |
| 5     | 0.2000       |
| 6     | 0.0000       |

0은 자식 없이 혼자 탑승한 성인을 가르킨다. 데이터 분석 결과 아이를 1~3명 데리고 있을 때 꾸준히 생존율이 올라갔다. 하지만 4명 이후 생존율은 급격히 낮아졌다.

아이는 1~3명을 가진 부모의 생존율이 높다.


```python
import pymysql
import matplotlib.pyplot as plt

# 데이터베이스 연결 설정
connection = pymysql.connect(host='localhost', user='user', password='password', db='titanic')

try:
    with connection.cursor() as cursor:
        sql = """SELECT
                  Parch,
                  AVG(Survived) AS SurvivalRate
                 FROM
                  train
                 GROUP BY
                  Parch;"""
        cursor.execute(sql)
        result = cursor.fetchall()

        parch_values = []
        survival_rates = []
        for row in result:
            parch_values.append(row[0])  # Parch 값은 첫 번째 칼럼
            survival_rates.append(row[1])  # SurvivalRate 값은 두 번째 칼럼

finally:
    connection.close()

plt.bar(parch_values, survival_rates)
plt.title('Survival Rate by Number of Parents/Children')
plt.xlabel('Number of Parents/Children')
plt.ylabel('Survival Rate')
plt.show()
```

![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/5664ebf7-e78d-42b8-a752-3681fa04490a)


### Fare 요금, Pclass, 생존율에 관한 분석 및 시각화

먼저 SQL 쿼리 작성

```sql
SELECT
  Pclass,
  AVG(Fare) AS AvgFare,
  AVG(Survived) AS SurvivalRate
FROM
  train
GROUP BY
  Pclass;
```

| Parch | Mean Survival Rate | Total Survival Rate |
|-------|-------------------|--------------------|
| 1     | 87.96%            | 0.6559             |
| 2     | 21.47%            | 0.4798             |
| 3     | 13.23%            | 0.2394             |


분석 결과로

PclASS1 > Pclass > Pclass 순으로 평균 요금이 더 높았다.



```python

import pymysql
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 데이터베이스 연결 설정
connection = pymysql.connect(host='localhost', user='user', password='password', db='titanic')

try:
    with connection.cursor() as cursor:
        sql = """SELECT
                  Pclass,
                  AVG(Fare) AS AvgFare,
                  AVG(Survived) AS SurvivalRate
                 FROM
                  train
                 GROUP BY
                  Pclass
                 ORDER BY
                  Pclass;""" # Pclass로 정렬
        cursor.execute(sql)
        result = cursor.fetchall()

        pclass = []
        avg_fare = []
        survival_rates = []
        for row in result:
            pclass.append(row[0])
            avg_fare.append(row[1])
            survival_rates.append(row[2])

finally:
    connection.close()

# 표준화와 정규화 수행
scaler_std = StandardScaler()
scaler_norm = MinMaxScaler()

avg_fare_std = scaler_std.fit_transform([[x] for x in avg_fare])
avg_fare_norm = scaler_norm.fit_transform([[x] for x in avg_fare])

survival_rates_std = scaler_std.fit_transform([[x] for x in survival_rates])
survival_rates_norm = scaler_norm.fit_transform([[x] for x in survival_rates])

# 표준화된 값으로 그래프 그리기
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.bar(pclass, avg_fare_std.ravel(), alpha=0.5, label='Standardized Average Fare')
ax2.plot(pclass, survival_rates_std.ravel(), color='red', label='Standardized Survival Rate')

ax1.set_title('Standardized Survival Rate and Average Fare by Pclass')
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Standardized Average Fare')
ax2.set_ylabel('Standardized Survival Rate')
plt.legend()
plt.show()

# 정규화된 값으로 그래프 그리기
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.bar(pclass, avg_fare_norm.ravel(), alpha=0.5, label='Normalized Average Fare')
ax2.plot(pclass, survival_rates_norm.ravel(), color='red', label='Normalized Survival Rate')

ax1.set_title('Normalized Survival Rate and Average Fare by Pclass')
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Normalized Average Fare')
ax2.set_ylabel('Normalized Survival Rate')
plt.legend()
plt.show()

```

![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/487a446a-771d-4e50-85b7-85126f46004b)


티켓 클래스에 따른 평균 요금과 생존율을 보여준다.




## 승선 장소별 생존율

```sql
SELECT
  Embarked,
  AVG(Survived) AS SurvivalRate
FROM
  train
GROUP BY
  Embarked
ORDER BY
  Embarked;
```


| Embarked | SurvivalRate |
|----------|--------------|
|    C     |    0.6077    |
|    S     |    0.3628    |
|    Q     |    0.2857    |

탑승 장소 C > S > Q 별로 생존율이 상이하게 다르다. 이 점은 생존율 예측에 참고할 만한 점인 것 같다.

```

파이썬 코드는 다음과 같다.

```python
import pymysql
import matplotlib.pyplot as plt

# 데이터베이스 연결 설정
connection = pymysql.connect(host='localhost', user='user', password='password', db='titanic')

try:
    with connection.cursor() as cursor:
        sql = """SELECT
                  Embarked,
                  AVG(Survived) AS SurvivalRate
                 FROM
                  train
                 GROUP BY
                  Embarked
                 ORDER BY
                  Embarked;"""
        cursor.execute(sql)
        result = cursor.fetchall()

        embarked_ports = []
        survival_rates = []
        for row in result:
            embarked_ports.append(row[0])
            survival_rates.append(row[1])

finally:
    connection.close()

plt.bar(embarked_ports, survival_rates, alpha=0.5, label='Survival Rate')
plt.title('Survival Rate by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Survival Rate')
plt.legend()
plt.show()
```

![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/ed1bf9b4-7227-45e3-9a9d-4b5d6bb48ac1)





## 정리

### 타이틀 별 생존율

- 'Mrs.': 78.70%
- 'Miss.': 71.92%
- 'Mr.': 16.83%
- 'Mrs.'와 'Mr.' 사이의 생존율 차이는 약 61.87%로, 'Mrs.'의 생존율이 'Mr.'보다 매우 높았습니다.

### 성별 별 생존율:

- 여성: 75.48%
- 남성: 20.53%
- 여성의 생존율은 남성보다 약 54.95% 높았습니다.

### 객실 등급별 생존률:

- 1등급: 65.59%
- 3등급: 23.94%
- 1등급과 3등급 객실 사이의 생존율 차이는 약 41.65%로, 높은 등급의 객실이 생존율에 중요한 영향을 미쳤습니다.

### 연령대 별 생존율:

- 0-9세: 61.29%
- 70-79세: 0%
- 어린이의 생존율이 가장 높았으며, 노인의 생존율이 가장 낮았습니다.

### 성별, 형제 및 자매가 있는 경우 생존율

- Female, SibSp(0) : 0.7986
- male, SibSp(0) : 0.1835
- 성별이 여성이고 자녀가 없거나 1명, 2명 있을 경우 생존율이 가장 높았으며, 남성인 경우 대체적으로 낮았습니다.


### Fare 요금, Pclass, 생존율에 관한 분석 및 시각화

- Pcalss 1 : 0.6559 > Plcass 2 : 0.4798, Pclass 3 : 0.2394
- 1 > 2 > 3 순으로 생존율이 높았다.

### 승선 장소별 생존율

- Embarked C : 0.6077 > Embarked S : 0.3628 > Embarked Q : 0.2857
- C > S > Q 탑승 장소별로 생존율이 높았다.
- 


자 이제 데이터 분석을 가지고 최종 산출물을 만들어 보겠습니다

머신러닝 모델 : 랜덤 포레스트 분류기, 그리드 검색을 통한 하이퍼 파라미터 튜닝


```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 데이터 불러오기
file_path = "your.csv"
data = pd.read_csv(file_path)

# 결측치 처리
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# 범주형 데이터 인코딩
label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])
data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

# 특성 선택 및 추가
X = data[['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# 훈련 세트와 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성
model = RandomForestClassifier()

# 하이퍼파라미터 튜닝
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 최적의 모델
best_model = grid_search.best_estimator_

# 예측
y_pred = best_model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print(X_train.shape, y_train.shape, X_test.shape)
X_train.head()
```


## 데이터 전처리:

- 결측치 처리: 'Age'의 결측값은 중앙값으로 대체, 'Embarked'의 결측값은 최빈값으로 대체
- 범주형 변수 인코딩: 'Sex', 'Embarked'
- 특성 선택: 'Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

데이터 분할: 훈련 세트 80%, 테스트 세트 20%

모델 선택: 랜덤 포레스트 분류기

### 하이퍼파라미터 튜닝:

- n_estimators: [100, 200]
- max_depth: [10, 20]
- min_samples_split: [2, 5]
- 모델 훈련: 최적의 하이퍼파라미터로 훈련

### 예측: 테스트 세트를 사용해 예측

### 성능 평가:

정확도: 약 0.8212 (82.12%)

아직 부족한 것이 많지만.. 캐글에 제출을 합니다.

![image](https://github.com/plintAn/MySQL-Pratice-Titanic/assets/124107186/fa08056f-93f1-4a69-a96d-4c93fcb31fe8)
































