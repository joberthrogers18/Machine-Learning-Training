import pandas as pd

base = pd.read_csv('census.csv')

# Separando atributos previsores e de previsão(classe) que será supervisionado
previsors = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Biblioteca responsavel por transforma string para numeros
# Já que os algoritimos não lidam bem com string
# Para cada string ele associa um número correspondente
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsors = LabelEncoder() 
# labels = labelencoder_previsors.fit_transform(previsors[:, 1]) // teste de mudança de string para numero

values_categoricals = [1, 3, 5, 6, 7, 8, 9, 13]
# Transformando as features que tem string em número e guardando em previsores

for i in values_categoricals:
    previsors[:, i] = labelencoder_previsors.fit_transform(previsors[:, i])

# fazendo a mesma coisa para classe
labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

# nomarlizando os valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsors = scaler.fit_transform(previsors)

# dividir em base de treino e teste
from sklearn.model_selection import train_test_split
previsors_train, previsors_test, classe_train, classe_test = train_test_split(previsors, classe, test_size=0.15, random_state=0)

from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(
        verbose=True,
        max_iter=1000,
        tol=0.000010,
        learning_rate="constant"
    )
classifier.fit(previsors_train, classe_train)
prevision = classifier.predict(previsors_test)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(prevision, classe_test)
matrix = confusion_matrix(prevision, classe_test)