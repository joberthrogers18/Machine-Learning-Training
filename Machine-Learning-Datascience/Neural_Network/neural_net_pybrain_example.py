from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

network = FeedForwardNetwork()

# 2 é a quantidade de neurônio para camada de entrada
EntryLayer = LinearLayer(2)

# camada oculta com 3 neurônios
HideLayer = SigmoidLayer(3)

# camada de saída com 1 neurônio
ExitLayer = SigmoidLayer(1)

# unidades de Bias com neurônios extras
bias1 = BiasUnit()
bias2 = BiasUnit()

# Adicionando os modulos a rede com as configurações iniciais
network.addModule(EntryLayer)
network.addModule(HideLayer)
network.addModule(ExitLayer)
network.addModule(bias1)
network.addModule(bias2)

# Conectando as camadas
EntryToHideLayer = FullConnection(EntryLayer, HideLayer)
HideToExitLayer = FullConnection(HideLayer, ExitLayer)
biasToHideLayer = FullConnection(bias1, HideLayer)
biasToExitLayer = FullConnection(bias2, ExitLayer)

# rede neural e suas estruturas iram ser criadas
network.sortModules()

print(network)
# printa os pesos para ligação das camadas
print(EntryToHideLayer.params)
