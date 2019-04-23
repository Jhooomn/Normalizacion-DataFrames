# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 08:11:27 2019

@author: Jhon Baron
"""
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn import preprocessing

#Distribucion uniforme en histograma de datos aleatorios
data=np.random.uniform(1, 5, 1000000)
plt.hist(data);

genero = ['Hombre','Mujer']
estrato = ['1','2','3']
lista_genero =[]
lista_estrato = []
edad = []
#Agrupamiento Hombre - Mujer
for i in range(0, 500):
    lista_genero.append(np.random.choice(genero))
    lista_estrato.append(np.random.choice(estrato))
   
df = pd.DataFrame({
        'Genero': lista_genero,
        'Estrato': lista_estrato,
        'Edad': 35 + 5*np.random.rand(500)
        })
    
#DataFrame Genero-Estrato-Edad    
print(df.head(5))
#Resumen de Datos Estadisticos
print()
est_resume=df.describe()
print(est_resume)
print()
#Agrupamiento
genero_agrupado = df.groupby('Genero');
for nombre, genero in genero_agrupado:
    print(nombre)
    print(genero)
print('DataFrame')
print(df)
print()

#Normalización Sklearn
scaler = preprocessing.Normalizer(norm='l2', copy=True)
df[['Estrato','Edad']]=scaler.fit_transform(df[['Estrato','Edad']])
print("*"*20)
print("Normalizacion Sklearn Normalizer")
print(df[['Estrato','Edad']])
print("*"*20)

#Normalizacion  Min Max
x = df[['Estrato','Edad']].values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
print("Max-Min")
print(df.sample(15)) #Mostrar solo 15


