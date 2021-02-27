#!/usr/bin/env python
# coding: utf-8

# # # 힙알마: 추천시스템 구현

# __콘텐츠 기반 필터링 : 사용자가 과거에 긍정적으로 평가한 콘텐츠와 유사한 콘텐츠를 추천__
# 
# ---
# 실습 주제: 영화 인셉션과 유사한 장르의 영화를 추천하기!
# 
# #### <데이터>
# 무비렌즈 영화 평점 데이터
# 
# 
# #### <중요 실습>
# 1. 영화 장르 TF-IDF 행렬로 벡터화하기
# 2. 코사인 유사도 계산 : TF-IDF 행렬로 변환한 데이터 세트를 코사인 유사도로 비교
# 
# ---

# ### Import Library (step.01)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.max_rows=150
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# ### Load Data (step.02)

# In[2]:


movie_file_path = 'D:/data/movie_recommendation/tmdb_5000_movies.csv'
movie_data = pd.read_csv(movie_file_path)
movie_df = movie_data.copy()

print(movie_df.shape)
movie_df.head()


# ### Data Processing (step.03)

# In[3]:


# 1. str to list / genres 컬럼의 'name' 만 가져오기
from ast import literal_eval
movie_df['genres_list'] = movie_df['genres'].apply(literal_eval)
movie_df['keywords_list'] = movie_df['keywords'].apply(literal_eval)

print(movie_df.shape)
print("type_genres_list: ",  type(movie_df['genres_list'][0]))
print("type_keywords_list: ",  type(movie_df['keywords_list'][0]))

# 2. return dict 'name' 
movie_df['genres_name'] = movie_df['genres_list'].apply(lambda x : [y['name'] for y in x])
movie_df['keywords_name'] = movie_df['keywords_list'].apply(lambda x : [y['name'] for y in x])

movie_df[['genres', 'genres_list', 'genres_name', 'keywords', 'keywords_list', 'keywords_name']].head()


# In[4]:


movie_df['genres_name'][0]


# ### TF-IDF(Term Frequency - Inverse Document Frequency) (step.04)
# 
# [자세한 설명: TF-IDF](https://wikidocs.net/31698)

# In[5]:


# TF-IDF

# 1. 문서를 토큰 리스트로 변환한다.
# 2. 각 문서에서 토큰의 출현 빈도를 센다.
# 3. 각 문서를 BOW(Bag of Words) 인코딩 벡터로 변환한다.

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
    'The last document?',
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)


# In[6]:


# 공백문자로 word 단위가 구분되는 문자열로 변환. 
movie_df['genres_literal'] = movie_df['genres_name'].apply(lambda x : (' ').join(x))
movie_df['genres_literal']


# In[7]:


# TF-IDF로 학습
count_vect = TfidfVectorizer(min_df=0, ngram_range=(1, 2)) #min_df: 단어장에 들어갈 최소빈도, ngram_range: 1 <= n <= 2
genre_mat = count_vect.fit_transform(movie_df['genres_literal'])
print(genre_mat.shape) 
# 4803 * 276의 크기를 가진 TF-IDF 행렬 생성
# 즉, 4803개 영화에 대한 276개 장르의 '장르 매트릭스'가 생성


# In[8]:


print(genre_mat[0])


# ### Movie Recommendation (step.05)

# #### Cosine_similarity?!!
# 코사인 유사도란 벡터와 벡터 간의 유사도를 비교할 때 두 벡터 간의 사잇각을 구해서 
# 
# 얼마나 유사한지 수치로 나타낸 것입니다.
# 
# 
# [자세한 설명: NLP - 8. 코사인 유사도(Cosine Similarity)](https://bkshin.tistory.com/entry/NLP-8-%EB%AC%B8%EC%84%9C-%EC%9C%A0%EC%82%AC%EB%8F%84-%EC%B8%A1%EC%A0%95-%EC%BD%94%EC%82%AC%EC%9D%B8-%EC%9C%A0%EC%82%AC%EB%8F%84)

# In[9]:


from sklearn.metrics.pairwise import cosine_similarity
# 영화별 장르 유사도 계산한 매트릭스 생성
genre_cosine_similarity = cosine_similarity(genre_mat, genre_mat)
print(genre_cosine_similarity.shape)
genre_cosine_similarity[0]


# In[10]:


# 자료를 정렬하는 것이 아니라 순서만 알고 싶다면 argsort
# 유사도가 높은 영화를 앞에서부터 순서대로 보여줌
# 0번째 영화의 경우 유사도 순서 : 0번, 3494번, 813번, ..., 2401 순서
genre_cosine_similarity_sorted_ind = genre_cosine_similarity.argsort()[:, ::-1] # 전체를 -1칸 간격으로
print(genre_cosine_similarity_sorted_ind[:1])


# In[11]:


# 영화별 장르를 기준으로 코사인 유사도에 따라 추천하기
def recommender_movie_genre_cosine_similarity(df, sorted_index, title_name, top_n = 10):
    
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values
    similar_index = sorted_index[title_index, :(top_n)]
    
    print(similar_index)
    
    similar_index = similar_index.reshape(-1)
    
    return df.iloc[similar_index]


# In[13]:


# test: Inception과 유사한 영화 10개 추천
recommemd_movie_test = recommender_movie_genre_cosine_similarity(movie_df, 
                                                                genre_cosine_similarity_sorted_ind,
                                                                'Inception',
                                                                10)

recommemd_movie_test[['title', 'vote_average', 'genres_name', 'vote_count']]


# In[ ]:




