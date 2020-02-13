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
from sklearn.preprocessing import LabelEncoder
labelencoder_previsors = LabelEncoder() 
# labels = labelencoder_previsors.fit_transform(previsors[:, 1]) // teste de mudança de string para numero

# Transformando as features que tem string em número e guardando em previsores
previsors[:, 1] = labelencoder_previsors.fit_transform(previsors[:, 1])
previsors[:, 3] = labelencoder_previsors.fit_transform(previsors[:, 3])
previsors[:, 5] = labelencoder_previsors.fit_transform(previsors[:, 5])
previsors[:, 6] = labelencoder_previsors.fit_transform(previsors[:, 6])
previsors[:, 7] = labelencoder_previsors.fit_transform(previsors[:, 7])
previsors[:, 8] = labelencoder_previsors.fit_transform(previsors[:, 8])
previsors[:, 9] = labelencoder_previsors.fit_transform(previsors[:, 9])
previsors[:, 13] = labelencoder_previsors.fit_transform(previsors[:, 13])
