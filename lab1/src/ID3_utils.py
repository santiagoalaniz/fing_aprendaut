# En este archivo python se encuentra la defincion de todas las funciones y metodos auxiliares
# para nuestra implementacion del algoritmo ID3

class ID3Node:
    def __init__(self, label, info_gain):
        self.label = label
        self.info_gain = info_gain
        self.children = {}

def node(label, info_gain= 1.): return ID3Node(label, info_gain)

def max_gain_attr(exs, attr_tget, attrs):
    return attrs[0], 0.

def evaluate(X, tree):
    return X.apply(lambda x: 0, axis=1)
