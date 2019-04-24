import numpy as np;
import pandas as pd;
from sklearn import preprocessing

data = pd.read_csv("movies.csv", header=0)
df = pd.DataFrame(data)
print("Data Frame: Movies")
print(df)

#Normalización Sklearn
scaler = preprocessing.Normalizer(norm='l2', copy=True)
df[['imdb_score','movie_facebook_likes']]=scaler.fit_transform(df[['imdb_score','movie_facebook_likes']])
print("*"*20)
print("Normalizacion Sklearn Normalizer")
print(df[['imdb_score','movie_facebook_likes']])
print("*"*20)

#Normalizacion  Min Max
x = df[['imdb_score','movie_facebook_likes']].values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
print("Max-Min")
print(df.sample(15)) #Mostrar solo 15




