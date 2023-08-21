# En este archivo python se encuentra la defincion de todas las funciones y metodos auxiliares
# para nuestra implementacion del algoritmo ID3

class ID3Node:
    def __init__(self, label, children, info_gain):
        self.label = label
        self.info_gain = info_gain
        self.children = children

def pos_root(): return ID3Node("root", True)

def neg_root(): return ID3Node("root", False)
