#!/usr/bin/env python
# coding: utf-8

# # 힙알마: 추천시스템 구현

# __협업 필터링 : 사용자와 유사한 취향을 가진 다른 사용자들과 공통적으로 좋아하는 콘텐츠를 추천__
# 
# __이웃기반협업필터링:두 사용자 혹은 콘텐츠가 공통적으로 가지고 있는 평가에 의해 정의된 콘텐츠 또는 사용자 사이의 유사성 지표에 의존__ 
# 
# ---
# 실습 주제: 영화 라이언 일병 구하기에 5점을 준 사용자가 좋아할 영화 추천
# 
# #### <데이터>
# 무비렌즈 영화 평점 데이터
# 
# 
# #### <중요 실습>
# 1. 사용자-아이템 평점 행렬로 변환
# 2. 코사인 유사도 계산 : 사용자-아이템 평점 행렬로 변환한 데이터 세트를 코사인 유사도로 비교
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


# 사용자들의 영화에 대한 평가 데이터
rating_file_path = 'D:/data/movie_recommendation/ratings.csv'
ratings = pd.read_csv(rating_file_path).drop('timestamp', axis = 1)
print(ratings.shape)
ratings.head()


# In[3]:


# 영화 제목 데이터
movie_file_path = 'D:/data/movie_recommendation/movies.csv'
movies = pd.read_csv(movie_file_path).drop('genres', axis = 1)
print(movies.shape)
movies.head()


# ### Data Processing (step.03)

# In[4]:


# train, test set 분리
from sklearn.model_selection import train_test_split
X = ratings.copy()
y = ratings['userId']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 91)


# In[5]:


# 사용자-아이템 평점 행렬로 변환
rating_matrix = X_train.pivot_table(index = 'userId', columns = 'movieId', values = 'rating')
rating_matrix


# ### Movie Recommendation (step.04)

# In[6]:


# 코사인 유사도 산출
from sklearn.metrics.pairwise import cosine_similarity

# NaN 값을 모두 0 으로 변환
rating_matrix_dummy = rating_matrix.copy().fillna(0)

# 사용자 간 코사인 유사도 산출
user_similarity = cosine_similarity(rating_matrix_dummy, rating_matrix_dummy)

# cosine_similarity() 로 반환된 넘파이 행렬을 영화명을 매핑하여 DataFrame으로 변환
user_similarity = pd.DataFrame(data = user_similarity, 
                               index = rating_matrix.index,
                               columns = rating_matrix.index)

print(user_similarity.shape)
user_similarity.head()


# In[7]:


# movieId의 가중 평균 rating 계산하는 함수
# 가중치는 주어진 사용자와 다른 사용자 간의 유사도(user_similarity)
# 주어진 userId와 movieId에 대해 cf 알고리즘으로 예상 평점 구하기 
def cf_user(userId, movieId):
    if movieId in rating_matrix:
        # userId와 다른 사용자 유사도 복사
        similarity_scores = user_similarity[userId].copy()
        
        # movieId에 대한 다른 사용자의 모든 평점을 복사
        movie_ratings = rating_matrix[movieId].copy()
        
        # movieId에 대해 평가하지 않은 사용자 index
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        
        # movieId에 대해 평가하지 않은 사용자의 평점 제거
        movie_ratings = movie_ratings.dropna()
        
        # movieId에 대해 평가하지 않은 사용자와의 유사도 제거
        similarity_scores = similarity_scores.drop(none_rating_idx)
        
        # movieId에 대해 평가한 각 사용자에 대해서 평점을 유사도로 가중평균한 예측 평점 구하기
        mean_rating = np.dot(similarity_scores, movie_ratings) / similarity_scores.sum()
    
    else:
        mean_rating = 2.5
        
    return mean_rating


# In[27]:


cf_user(1, 2028)


# In[9]:


# 추천 정확도(RMSE) 계산

# from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

# rmse score
def score(model):
    id_pairs = zip(X_test['userId'], X_test['movieId'])
    y_pred = np.array([model(user, movie) for (user,movie) in id_pairs])
    y_true = np.array(X_test['rating'])
    return rmse(y_true, y_pred)


# In[17]:


print('RMSE SCORE: ', score(cf_user))


# ### 특정 사용자에게 추천하기

# In[11]:


# 라이언 일병 구하기 movieId 찾기
movies[movies['title'] == 'Saving Private Ryan (1998)']


# In[12]:


# 라이언 일병 구하기에 5점을 준 사용자 찾기
ratings[(ratings['movieId'] == 2028) & (ratings['rating'] == 5.0)].head()


# In[13]:


rating_matrix = ratings.pivot_table(index = 'userId', columns = 'movieId', values = 'rating')
rating_matrix


# In[14]:


# 전체 데이터로 코사인 유사도 구하기
from sklearn.metrics.pairwise import cosine_similarity

rating_matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(rating_matrix_dummy, rating_matrix_dummy)
user_similarity = pd.DataFrame(data = user_similarity, 
                               index = rating_matrix.index,
                               columns = rating_matrix.index)
print(user_similarity.shape)
user_similarity.head()


# In[29]:


user_movie = rating_matrix.loc[33].copy()
user_movie


# In[23]:


# 영화 추천하기
def recommendation_movie(user_id, n_items):
    user_movie = rating_matrix.loc[user_id].copy()
    for movie in rating_matrix:
        if pd.notnull(user_movie.loc[movie]):
            user_movie.loc[movie] = 0
    movie_sort = user_movie.sort_values(ascending = False)[:n_items]
    recom_movies = movies.loc[movie_sort.index]
    recommendations = recom_movies['title']
    return recommendations


# In[28]:


# 라이언 일병 구하기 영화에 5점을 준 11번 사용자에게 영화 10편 추천하기
recommendation_movie(33, 10)


# In[ ]:




