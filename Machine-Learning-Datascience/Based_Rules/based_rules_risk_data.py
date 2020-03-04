import Orange

base = Orange.data.Table('../pre-processing/risco-credito.csv')
base.domain

# usando algoritmo cn2
# Algoritimo que vai fazer a classificação das regras
cn2_learner = Orange.classification.rules.CN2Learner()
# regras em si
# não encontra a classe de previsão direto logo é necessário colocar c# na frente da label do dataset para ele reconhecer
# No Orange geralmente isso é tudo manual colocando no dataset mesmo, basta olhar a documentação
classifier = cn2_learner(base)

# regras geradas
for rule in classifier.rule_list:
    print(rule)
    
# 1º - Historia: Boa, Divida: Alta, Garantia: Nenhuma, Renda: > 35
# 2º - Historia: Ruim, Divida: Alta, Garantia: Adequada, Renda: < 15
    
result = classifier([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])

for i in result:
    print(base.domain.class_var.values[i])