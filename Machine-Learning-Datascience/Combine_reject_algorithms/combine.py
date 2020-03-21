import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Carregando arquivo
svm = pickle.load(open('../Classifiers_save/svm_finished.sav', 'rb'))
neural_network = pickle.load(open('../Classifiers_save/Neural_Network_finished.sav', 'rb'))
random_forest = pickle.load(open('../Classifiers_save/random_forest_finished.sav', 'rb'))

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

pay = 0
not_pay = 0

if response_svm[0] == 1:
    pay += 1
else:
    not_pay += 1
    
if response_random_forest[0] == 1:
    pay += 1
else:
    not_pay += 1

if response_NN[0] == 1:
    pay += 1
else:
    not_pay += 1
    
if pay > not_pay:
    print('The client will pay')
elif pay == not_pay:
    print('Draw')
else:
    print('The client will NOT pay')