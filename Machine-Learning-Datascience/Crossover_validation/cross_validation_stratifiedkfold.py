
import pandas as pd 

base = pd.read_csv('credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92

previsors = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
previsors[:, 1:4] = imputer.fit_transform(previsors[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsors = scaler.fit_transform(previsors)

from sklearn.naive_bayes import GaussianNB

a = np.zeros(5)
# formato da vairável
previsors.shape

# criando um vetor de zeros
b = np.zeros(shape=(previsors.shape[0], 1 ))

# organizar as classes de forma que cada um dos conjuntos
# tenha metade das intancias para cada classes previsora, quantidade proporcional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state= 0)
results  = []
result_accuracy = []
matrizes = []
for indice_train, indice_test in kfold.split(previsors,
                                             np.zeros(shape=(previsors.shape[0], 1 ))):
    #print('Indice training ', indice_train, ' Indice test ', indice_test)
    classifier = GaussianNB()
    classifier.fit(previsors[indice_train], classe[indice_train])
    previsions = classifier.predict(previsors[indice_test])
    results.append({'acc': accuracy_score(previsions, classe[indice_test]), 'matrix': confusion_matrix(previsions, classe[indice_test])})
    result_accuracy.append(accuracy_score(previsions, classe[indice_test]))
    matrizes.append(confusion_matrix(previsions, classe[indice_test]))

# fazendo media de todas as matrizes de confusão
# axis 0 é para pegar por linha, sem isso ele não consegue separar as colunas
matrix_final = np.mean(matrizes, axis=0) 

result_accuracy = np.asarray(result_accuracy)
result_accuracy.mean()
result_accuracy.std()