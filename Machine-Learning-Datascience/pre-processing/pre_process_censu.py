#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:48:12 2020

@author: joberth rogers
"""

import pandas as pd

base = pd.read_csv('census.csv')

# Separando atributos previsores e de previsão(classe) que será supervisionado
previsors = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Biblioteca responsavel por transforma string para numeros
# Já que os algoritimos não lidam bem com string
# Para cada string ele associa um número correspondente
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsors = LabelEncoder() 
# labels = labelencoder_previsors.fit_transform(previsors[:, 1]) // teste de mudança de string para numero

values_categoricals = [1, 3, 5, 6, 7, 8, 9, 13]
# Transformando as features que tem string em número e guardando em previsores

for i in values_categoricals:
    previsors[:, i] = labelencoder_previsors.fit_transform(previsors[:, i])

# Usando estratégia "dummy" para separar vaiaveis que não tem carater ordinal 
# em mais variaveis com cada um dos conteúdos, informando se contém ou não
# um determinado atributo na tabela
# Ex: white | Black | Calcasian
#      0        1       0
#      1        1       1 
# previsors = base.iloc[:, 8:9].values
# previsors[:, 0] = labelencoder_previsors.fit_transform(previsors[:, 0])   
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsors = onehotencorder.fit_transform(previsors).toarray()

# fazendo a mesma coisa para classe
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

# nomarlizando os valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsors = scaler.fit_transform(previsors)