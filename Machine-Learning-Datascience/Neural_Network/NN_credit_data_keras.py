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
previsors_train, previsors_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.25, random_state=0)

import keras
# modelo sequencial feed foward
from keras.models import Sequential
# Dense é o tipo de rede que se conecta o neurônios com todas outras camadas
# a rede neural convencional
from keras.layers import Dense

classifier = Sequential()
# Adicionando uma novas camada ocultas
#units são a quantidade de neurônios na camada
# input_dim é a quantidade de entradas da base e só vai para a primeira camada oculta
classifier.add(Dense(units=2, activation='relu', input_dim=3))
classifier.add(Dense(units=2, activation='relu'))
#camada de saída
# para modelos de mais de uma camada oculta para função de ativação é melhor a softmax, no caso de um usa a sigmoide
classifier.add(Dense(units=1, activation='sigmoid'))
# compular o classificador
# adam é uma descida de gradiente com uma gap
# loss é a função de erro usada para calcular o erro
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# treinamento
# batch size seria o gap para a mini descidada de gradiente ajustar os pesos
# nb_epoch é o número de epocas que o algoritmo vai usar para achar a melhor forma de classificação
classifier.fit(previsors_train, classe_train, batch_size = 10, nb_epoch= 100)
previsions = classifier.predict(previsors_test)

# transformando as previsões para as classes preditas em vez do valor dado pela função sigmoide
# 0.5 é um limiar que é definido por mim para classificar entre as classes preditas
previsions = (previsions > 0.5)

from sklearn.metrics import accuracy_score, confusion_matrix
precision = accuracy_score(previsions, classe_test)
matrix = confusion_matrix(previsions, classe_test)
