import Orange

# Não precisa fazer os pre processamentos dos algoritimos anteriores pq o Orange já lida com isso
base = Orange.data.Table('../pre-processing/credit-data.csv')

#                                classe preditiva 
# [clientid, income, age, loan | default]
base.domain

# base dividida em 75% | 25%
divide_base = Orange.evaluation.testing.sample(base, n=0.25)
test_base = divide_base[0]
train_base = divide_base[1]

# gerando as regras
cn2_learner = Orange.classification.rules.CN2Learner()
classifier = cn2_learner(base)

# Visualizando as regras
for rule in classifier.rule_list:
    print(rule)

# verificando o quão bom ficou a classificação pelas regras
result = Orange.evaluation.testing.TestOnTestData(train_base, test_base, [classifier])

# Acuracia, precisão
print(result)