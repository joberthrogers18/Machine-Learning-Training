#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:34:52 2020

@author: jobs
"""

import pandas as pd

base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:, :4 ].values
classes = base.iloc[:, 4 ].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Transformando atributos do tipo string para inteiros, pois o Naive Bayers não aceita atributos do nominal
for i in range(4):
    previsores[:, i] = labelencoder.fit_transform(previsores[:, i])
    
# Não é necessário nesse caso fazer o Labelenconder para o tipo de classes

# Classe do algoritimo Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
# Gerando a tabela de probabilidade do algoritimo de Naive Bayes
# ---------------------TREINAMENTO ---------------------------------
classificador.fit(previsores, classes)

# --------------------------TESTE -------------------------------------
# 1º - Historia: Boa, Divida: Alta, Garantia: Nenhuma, Renda: > 35
# 2º - Historia: Ruim, Divida: Alta, Garantia: Adequada, Renda: < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])

# ver quais as classes disponiveis para ser classificado
print(classificador.classes_)
#contagem das classes
print(classificador.class_count_)
# probabilidade apriori, que seria cada valor do class count dividido pelo total 
print(classificador.class_prior_)