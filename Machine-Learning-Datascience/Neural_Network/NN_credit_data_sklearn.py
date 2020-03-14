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

# importando multilayer perceptron 
from sklearn.neural_network import MLPClassifier
# verbose deixa logs na hora do treinamento
# max_iter é o total de iterações, por default é 200
# tol é a tolerancia
classifier = MLPClassifier(
        verbose=True,
        max_iter=1000,
        tol=0.000010,
        solver='adam',
        hidden_layer_sizes=(100,100),
        activation='relu'
    )
classifier.fit(previsors_train, classe_train)
precision = classifier.predict(previsors_test)

from sklearn.metrics import accuracy_score, confusion_matrix
precision = accuracy_score(precision, classe_test)
matrix = confusion_matrix(precision, classe_test)
