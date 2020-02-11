#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:53:46 2020

@author: joberth rogers
"""

import pandas as pd
base = pd.read_csv('credit-data.csv')

# informaçoes da base
# base.describe()

# atribuindo valores negativos de idade a media das idades 
# para tirar a inconsitencia
base.loc[ base.age < 0 ] = base.age[base.age > 0].mean()
