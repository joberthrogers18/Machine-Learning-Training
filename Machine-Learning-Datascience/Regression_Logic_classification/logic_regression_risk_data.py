import pandas as pd

base = pd.read_csv('risk_credit_regre_logic.csv')
previsores = base.iloc[:, :4 ].values
classes = base.iloc[:, 4 ].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Transformando atributos do tipo string para inteiros, pois o Naive Bayers não aceita atributos do nominal
for i in range(4):
    previsores[:, i] = labelencoder.fit_transform(previsores[:, i])
    
from sklearn.linear_model import LogisticRegression

classificador = LogisticRegression()
classificador.fit(previsores, classes)

# Valor do parametro B0, gerado pelo treinamento de regressão logistica
# onde a reta intercepta o eixo x
print(classificador.intercept_)

# coeficientes para cada atributo, seria o (B1, B2, B3...)
# y = B + B1*historia + B2*divida + B3*garantia + B4 venda
print(classificador.coef_)

# --------------------------TESTE -------------------------------------
# 1º - Historia: Boa, Divida: Alta, Garantia: Nenhuma, Renda: > 35
# 2º - Historia: Ruim, Divida: Alta, Garantia: Adequada, Renda: < 15
resultado = classificador.predict([[0,0,1,2], [3, 0, 0, 0]])

# verificando as probabilidades em vez de apenas 'baixo' e 'alto'
resultado2 = classificador.predict_proba([[0,0,1,2], [3, 0, 0, 0]])

print(resultado)
print(resultado2)