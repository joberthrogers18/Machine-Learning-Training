import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('../pre-processing/credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92
previsors = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
previsors[:, 1:4] = impute.fit_transform(previsors[:, 1:4])
scaler = StandardScaler()
previsors = scaler.fit_transform(previsors)

# Carregando arquivo
svm = pickle.load(open('svm_finished.sav', 'rb'))
neural_network = pickle.load(open('Neural_Network_finished.sav', 'rb'))
random_forest = pickle.load(open('random_forest_finished.sav', 'rb'))

# fazendo a predição e calculando a acurácia direto
result_svm = svm.score(previsors, classe)
result_random_forest = random_forest.score(previsors, classe)
result_neural_network = neural_network.score(previsors, classe)

new_register = [[50000, 40, 5000]]
# formato que o classificador lê, em forma de array
new_register = np.asarray(new_register)
# -1 é quando eu não quero mexer naquele atributo
# quero apenas mexer com as colunas
# pois seu não trocar a forma da linhas e colunas, na hora de escalonar ele zera todos os valores
new_register = new_register.reshape(-1, 1)
# escalonando o registro
new_register = scaler.fit_transform(new_register)
new_register = new_register.reshape(-1, 3)

# predição
response_svm = svm.predict(new_register)
response_random_forest =  random_forest.predict(new_register)
response_NN =  neural_network.predict(new_register)