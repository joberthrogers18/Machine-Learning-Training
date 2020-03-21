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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

classifierSVM = SVC(kernel='rbf', C = 2.0, probability=True)
classifierSVM.fit(previsores, classe)

classifierRandomForest = RandomForestClassifier(n_estimators=40, criterion='entropy')
classifierRandomForest.fit(previsores, classe)

classifierMLP = MLPClassifier(verbose=True, 
                              max_iter=1000, 
                              tol=0.00001, 
                              solver='adam', 
                              hidden_layer_sizes=(100), 
                              activation='relu', 
                              batch_size=200,
                              learning_rate_init=0.001)
classifierMLP.fit(previsores, classe)

# salvando os modelos
# pickle salva arquivos em disco
import pickle
pickle.dump(classifierSVM, open('svm_finished.sav', 'wb'))
pickle.dump(classifierRandomForest, open('random_forest_finished.sav', 'wb'))
pickle.dump(classifierMLP, open('Neural_Network_finished.sav', 'wb'))
