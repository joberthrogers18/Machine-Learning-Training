from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

'''
# 2 neuronios para camada de entrada, 3 para oculta e 1 para saída
network = buildNetwork(2, 3, 1, 
                       ouclass=SoftmaxLayer,
                       hiddenclass=SigmoidLayer,
                       bias=False
                      )
print(network['in'])
print(network['hidden0'])
print(network['out'])
print(network['bias'])
'''

# 2 neuronios para camada de entrada, 3 para oculta e 1 para saída
network = buildNetwork(2, 3, 1)

# 2 atributos previsores e uma classe
base = SupervisedDataSet(2, 1)

# adicionando a base 
# exemplo do XOR
base.addSample((0,0), (0,))
base.addSample((0,1), (1,))
base.addSample((1,0), (1,))
base.addSample((1,1), (0,))

# print(base['input'])
# print(base['target'])

# treinamento usando taxa de aprendizado e momento para definir o menor erro possível
training = BackpropTrainer(
        network, 
        dataset=base, 
        learningrate=0.01, 
        momentum=0.06
    )

# epocas de treinamento: 30000
for i in range(1, 30000):
    error = training.train()
    if i % 1000 == 0:
        print("Error: %s" %(error))
        
# seria o predict
print(network.activate([0, 0]))
print(network.activate([0, 1]))
print(network.activate([1, 0]))
print(network.activate([1, 1]))