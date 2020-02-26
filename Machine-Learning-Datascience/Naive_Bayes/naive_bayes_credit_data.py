#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:12:46 2020

@author: Joberth Rogers
"""

import pandas as pd
import numpy as np

# Trocando idade negativa pela media 40,92
base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
# pegando previsores
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Tirando atributos NaN e substituindo pela media
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# escalonano valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# dividir em base de treino e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Usando Naive Bayes
from sklearn.naive_bayes import GaussianNB
modelo = GaussianNB()
modelo = modelo.fit(previsores_treinamento, classe_treinamento)

previsoes = modelo.predict(previsores_teste)

# verificando a precisão
from sklearn.metrics import confusion_matrix,accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)