import Orange

base = Orange.data.Table('../pre-processing/credit-data.csv')
base.domain

divide_base = Orange.evaluation.testing.sample(base, n=0.25)
test_base = divide_base[0]
train_base = divide_base[1]

# classifica pela maioria, não é um learner ele faz um count e classifica pela maioria
# algoritimo base
classifier = Orange.classification.MajorityLearner()
result = Orange.evaluation.testing.TestOnTestData(train_base, test_base, [classifier])

print(Orange.evaluation.CA(result))

from collections import Counter
print(Counter(str(d.get_class()) for d in test_base ))

# Counter({'0': 433, '1': 67})
# Como a maioria pertence a classe zero ele classifica como todos como sendo ela