---
title: "영화 추천 알고리즘"
date: 2020-06-03
draft: false

# post thumb
image: "images/featured-post/post4/post4.png"

# meta description
description: "this is meta description"

# taxonomies
categories: 
  - "Data analysis _Python"
tags:
  - "추천알고리즘"
  - "영화추천알고리즘"
  - "Python"
  - "kaggle"


# post type
type: "featured"
---

# Movies Recommender System

Package를 설치하는 환경폴더(my_env)를 따로 만들어 실행할 때마다 설치할 필요 없도록 설정했습니다.
이와 관련한 포스팅은 다음에 하겠습니다.

```python
import os, sys
from google.colab import drive
drive.mount('/content/drive')

my_path = '/content/notebooks'
os.symlink('/content/drive/My Drive/Colab Notebooks/my_env', my_path)
sys.path.insert(0, my_path)
```

```python
!pip install scikit-surprise
```

<hr>
저희가 사용할 모듈을 import 해주세요.

```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold

import warnings; warnings.simplefilter('ignore')
```

<hr>

## Simple Recommender

영화에 대한 다양한 정보들이 담긴 'movies_metadata.csv' 파일을 불러와 md 변수에 저장합니다

```python
md = pd.read_csv('/content/drive/My Drive/Colab Notebooks/MovieRcmmd/data/data/movies_metadata.csv')
md.head()
```

'genres'(장르) 컬럼의 각 행들의 타입이 dict. 장르 id와 이름으로 구성.

{'id': ... , 'name': '...'},{'id': ... ,'name' : '...'}, ,,,

```python
md['genres'][0] 
```

```python
print(type(md['genres'][0])) # "[{dict}, {dict},] <- dict들이 str으로 묶여있음
md['genres'].apply(literal_eval) # [{dict},{dict}] <- str을 list로 바꿔줌
```

```python
#각 영화 장르의 이름을 추출해 list 시켜줌
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x : [i['name'] for i in x] if isinstance(x, list) else [])
```

```python
md['genres']
```

```python
# eval(expression, globals=None, locals=none), 내장함수
# expression 인자에 string 값을 넣으면 해당 값을 그대로 실행하여 결과를 출력
expr = "10 + 10"
print(type(expr))
print(eval(expr))
```

```python
#ast.literal_eval(node_or_string), AST module에서 제공하는 함수
#문자 그대로 evaluate 실행하는 함수
import ast
str_dict = "{'a':3,'b':5}"
print(type(str_dict))
convert_dict= ast.literal_eval(str_dict)
print(convert_dict['a'])
print(convert_dict['b'])
```

```python
#literal_eval은 ValueError를 발생시킬 수 있다.
ast.literal_eval("10*2")
#literal_eval은 python의 기본 자료형 정도만 evaluate가 가능하도록 지원. eval과 비교해 훨씬 엄격하기 때문에 결과적으로 안전을 보장.
```

```python
type(md['vote_count'][0]) #vote_count의 타입이 float
```

```python
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int') #float을 int로
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C
```

```python
m = vote_counts.quantile(0.95)
m
```

```python
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x : str(x).split('-')[0] if x != np.nan else np.nan)
# error='coerce'
# 날짜로 된 문자열이 아니라 문자로 된 문자열인 경우 파싱할 수 없다는 ValueError가 발생하는 것을 방지하기 위해 'coerce'옵션 추가
# 파싱할 수 없는 문자열은 'NaN'으로 강제 변환
```

```python
md['year']
```

```python
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
qualified.shape #(행 수, 열 수)
```

```python
def weighted_rating(x):
  v = x['vote_count']
  R = x['vote_average']
  return (v/(v+m)*R)+(m/(m+v)*C) # Weighted Rating(WR)
# v is the number of votes for the movie
# m is the minimum votes required to be listed in the chart
# R is the average rating of the movie
# C is the mean vote across the whole report
```

```python
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
```

```python
qualified = qualified.sort_values('wr', ascending=False).head(250)
```

```python
qualified['wr']
```

```python
qualified.head(15)
```

```python
md.apply(lambda x : pd.Series(x['genres']), axis=1)
```

```python
md.apply(lambda x : pd.Series(x['genres']), axis=1).stack()
```

```python
s = md.apply(lambda x : pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
# 인덱스 1단계 제거(첫번째 열)
s
```

```python
s.name = 'genre'
gen_md = md.drop('genres', axis = 1).join(s)
gen_md
```

```python
def build_chart(genre, percentile=0.86):
  df = gen_md[gen_md['genre'] == genre]
  vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
  vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
  C = vote_averages.mean()
  m = vote_counts.quantile(percentile)

# 여러 개의 칼럼 데이터 추추 시 대괄호 두번[[]]
  qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
  qualified['vote_count'] = qualified['vote_count'].astype('int')
  qualified['vote_average'] = qualified['vote_average'].astype('int')

  qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m)*x['vote_average'])+(m/(m+x['vote_count'])*C), axis=1)
  qualified = qualified.sort_values('wr', ascending=False).head(250)

  return qualified
```

Top Romance Movies

```python
build_chart('Romance').head(15)
```

```python
links_small = pd.read_csv('/content/drive/My Drive/Colab Notebooks/MovieRcmmd/data/data/links_small.csv')
links_small
# imdbid : internet movie data base
# tmdbid : the movie data base
```

```python
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
links_small
```

```python
# 여러 개의 행 추출 시 대괄호 두 번 [[]]
md.loc[[19730, 29503, 35587]] #id가 날짜 형식
```

```python
md = md.drop([19730, 29503, 35587])
```

```python
md['id'] = md['id'].astype('int')
```

```python
smd = md[md['id'].isin(links_small)]
smd.shape
```

```python
smd.loc[1]['title']
```

```python
smd.loc[1]['overview'] # overview : 개요
```

```python
smd.loc[1]['tagline'] # tagline : 광고 구호, 슬로건
```

```python
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
```

```python
# tf-idf 인코딩 : 단어 갯수 그대로 카운트하지 않고 모든 문서에 공통적으로 들어있는 단어의 경우 문서 구별 능력이 떨어진다고 보아 가중치를 축소하는 방법
# analyzer='word'/'char'/'char_wb'(단어 내의 문자)
# ngram_range=(min_n, max_n)
# min_df=[0.0,1.0]사이의 실수. 디폴트 1, 단어장에 포함되기 위한 최소 빈도
# stop_words='english' : 영어용 스탑 워드 사용
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
```

```python
tf.vocabulary_ # 단어를 인덱싱, type은 dict
```

```python
tfidf_matrix.shape # 9099개의 문장이 268124 토큰으로 표현됨
```

```python
print(np.array(tfidf_matrix)) # (m, n) xxx : 문장 m에서의 단어 n의 tfidf값
```

```python
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

```python
cosine_sim[0]
# 1 :첫번째 문장 자기자신과의 코사인 유사도
# 0.00680476 :첫번째 문장과 두 번째 문장과의 코사인 유사도
```

```python
cosine_sim[9098]
```

```python
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
```

```python
smd
```

```python
indices
```

```python
def get_recommendations(title):
  idx = indices[title] #해당 title 인덱스
  sim_scores = list(enumerate(cosine_sim[idx])) #해당 title 과 다른 영화와의 유사도 인덱싱
  # (1, 0.00282712634654008) :해당 title과 1번째 영화와의 유사도가 0.002827126346...
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #유사도를 기준으로 내림차순으로 정렬
  sim_scores = sim_scores[1:31] #자기자신과의 유사도(1)을 제외하고, 유사도가 가장 높은 것부터 30개 추출
  movie_indices = [i[0] for i in sim_scores] #유사도 상위30개 영화에 대한 인덱스
  return titles.iloc[movie_indices]
```

```python
list(enumerate(cosine_sim[indices['The Godfather']]))
```

```python
sorted(list(enumerate(cosine_sim[indices['The Godfather']])), key=lambda x: x[1], reverse=True)
```

```python
get_recommendations('The Godfather').head(10)
```

```python
get_recommendations('The Dark Knight').head(10)
```

Metadata Based Recommender

```python
credits = pd.read_csv('/content/drive/My Drive/Colab Notebooks/MovieRcmmd/data/data/credits.csv')
keywords = pd.read_csv('/content/drive/My Drive/Colab Notebooks/MovieRcmmd/data/data/keywords.csv')
```

```python
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
```

```python
keywords.head() # 각 영화의 keywords 정리
```

```python
credits.head()
# cast :주연들에 대한 설명(character, name, gender,,)
# crew : 감독, 작가 등에 대한 설명
```

```python
md.shape
```

```python
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
md.head()
```

```python
smd = md[md['id'].isin(links_small)]
smd.shape
```

```python

```

```python
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
```

```python
def get_director(x):
  for i in x:
    if i['job'] == 'Director':
      return i['name']
  return np.nan
```

```python
smd['director'] = smd['crew'].apply(get_director)
```

```python
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x) #주연이름 세자리까지만
```

```python
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
```

```python
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ","")) for i in x]) # 주연이름 붙여쓰기, 소문자로
```

```python
smd['cast'].head()
```

```python
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x, x, x])
```

```python
smd['director']
```

Keywords

```python
s = smd.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s
```

```python
s = s.value_counts()
s[:5]
```

```python
s = s[s > 1]
```

```python
stemmer = SnowballStemmer('english') #접사 제거, 명사 추출
stemmer.stem('dogs')
```

```python
def filter_keywords(x):
  words = []
  for i in x:
    if i in s:
      words.append(i)
  return words
```

```python
smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords']
```

```python
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
```

```python
smd['keywords']
```

```python
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
smd['soup']
```

```python
count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
```

```python
count_matrix.shape # 9212개의 문장이 100304 토큰으로 표현
```

```python
print(np.array(count_matrix))
```

```python
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim
```

```python
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
indices
```

```python
get_recommendations('The Dark Knight').head(10)
```

```python
get_recommendations('Mean Girls').head(10)
```

Popularity and Ratings

```python
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified
```

```python
improved_recommendations('The Dark Knight')
```

```python
improved_recommendations('Mean Girls')
```

Collaborative Filtering

```python
reader = Reader()
```

```python
ratings = pd.read_csv('/content/drive/My Drive/Colab Notebooks/MovieRcmmd/data/data/ratings_small.csv')
ratings.head()
```

```python
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader) #데이터셋 로딩
svd = SVD() # SVD :특이값 분해 알고리즘

# 데이터를 5개의 부분집합{x1, x2, ,,, , x5}으로 나눈다.
# 5개의 부분집합 중 하나의 검증용 데이터셋(test_set)를 제외한 나머지 데이터셋을 학습용 데이터(train_set)로 사용하여 회귀분석 모형을 만들고 test_set으로 검증
# 5회 반복
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# kf = KFold(n_splits = 5)
# for train_set, test_set in kf.split(data):
#     svd.fit(train_set)
#     predictions = svd.test(test_set)
#     RMSE = accuracy.rmse(predictions, verbose=True)
#     MAE = accuracy.mae(predictions, verbose=True)
```

```python
print(svd.predict(1, 302))
# 1번 user가 302번 movie에 줬을 점수 추정치가 2.94
```

Hybrid Recommender

*   Input :User ID and the Title of a Movie
*   Output :Similar movies sorted on the basis of expected ratins by that
particular user


```python
def convert_int(x):
  try:
    return int(x)
  except:
    return np.nan
```

```python
id_map = pd.read_csv('/content/drive/My Drive/Colab Notebooks/MovieRcmmd/data/data/links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
id_mapindices
```

```python
indices_map = id_map.set_index('id')
indices_map
```

```python
def hybrid(userId, title):
  idx = indices[title]
  tmdbId = id_map.loc[title]['id']
  movie_id = id_map.loc[title]['movieId']
  sim_scores = list(enumerate(cosine_sim[int(idx)]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:26]
  movie_indices = [i[0] for i in sim_scores]
  movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
  movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
  movies = movies.sort_values('est', ascending=False)
  return movies.head(10)
```

```python
hybrid(1, 'Avatar')
# 1번 유저가 선택한 영화 'Avatar'와 유사도가 높은 25개의 영화를 추출해 젔을 법한 점수(rating) 추정
# rating est 상위 10개 추출
```

```python
hybrid(500, 'Avatar')
```
