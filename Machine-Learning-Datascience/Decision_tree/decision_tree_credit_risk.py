#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 22:34:52 2020

@author: Joberth
"""

import pandas as pd

base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:, :4 ].values
classes = base.iloc[:, 4 ].values


# Não consigo rodar uma arvore de decisão se tiver atributos categoricos, preciso transformar para números
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Transformando atributos do tipo string para inteiros, pois o Naive Bayers não aceita atributos do nominal
for i in range(4):
    previsores[:, i] = labelencoder.fit_transform(previsores[:, i])
    
# Não é necessário nesse caso fazer o Labelenconder para o tipo de classes

# Classe do algoritimo Arvore de decisão
from sklearn.tree import DecisionTreeClassifier, export
# escolhendo a entropia para verificar qual dos atributos devem estar no root
classificador = DecisionTreeClassifier(criterion='entropy')
# Gerando a arvore de decisão
# ---------------------TREINAMENTO ---------------------------------
classificador.fit(previsores, classes)

# Vendo as importancias das features (qual deve ficar mais acima na arvore por ser mais importante)
# segue a mesma ordem que está disposto no arquivo risco-credito.csv
print(classificador.feature_importances_)

# mostrando a arvore com arquivo .dot, é necessário ter o graphviz instalado
export.export_graphviz(classificador,
                       out_file='arvore.dot',
                       feature_names=['historia', 'divida', 'garantias', 'renda'],
                       class_names=['alto', 'baixo', 'moderado'],
                       filled=True,
                       leaves_parallel=True)

# --------------------------TESTE -------------------------------------
# 1º - Historia: Boa, Divida: Alta, Garantia: Nenhuma, Renda: > 35
# 2º - Historia: Ruim, Divida: Alta, Garantia: Adequada, Renda: < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])

# ver quais as classes disponiveis para ser classificado
print(classificador.classes_)