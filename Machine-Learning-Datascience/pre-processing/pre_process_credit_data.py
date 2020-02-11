#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:53:46 2020

@author: joberth rogers
"""

import pandas as pd
import numpy as np
base = pd.read_csv('credit-data.csv')

# informaçoes da base
# base.describe()

# atribuindo valores negativos de idade a media das idades 
# para tirar a inconsitencia
base.loc[ base.age < 0 ] = base.age[base.age > 0].mean()

#buscando valores Nulos na coluna age
base.loc[pd.isnull(base['age'])]


# separando os atributos previsores e atributos meta
# previsores e o restultado classificado
previsors = base.iloc[: ,1:4].values
meta = base.iloc[: , 4].values

# preencher valores faltantes pelo sktlearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# atualizando a base com esses atributos ruins descartados
imputer = imputer.fit(previsors[:, 0:3])
previsors[:, 0:3] = imputer.transform(previsors[:, 0:3])