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

# fazer validação cruzada
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()
# rodou dez vezes  dividindo a base em 10 partes e permutou isso
resultado = cross_val_score(classificador, previsores, classe, cv=10)
resultado.mean()
# desvio padrão
resultado.std()

# acerto médio é 92%
# validação cruzada é mais eficiente

