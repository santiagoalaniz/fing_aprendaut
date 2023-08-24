# G02 Laboratorio 01, Arboles de Decision.
# En este modulo de python se encuentran todas las funciones auxiliares que usamos
# para implementar ID3. Leer el informe para mas detalles sobre su funcionamiento.

import numpy as np

class ID3Node:
    def __init__(self, label, info_gain):
        self.label = label
        self.info_gain = info_gain
        self.children = {}

def node(label, info_gain= 1.): return ID3Node(label, info_gain)

def entropy(df, attr_tget):
    if df.empty: return 0.
    
    target_counts = df[attr_tget].value_counts(normalize=True)

    return -sum(p * np.log2(p) for p in target_counts if p > 0)

def max_gain_attr(df, attr_tget, attrs_values):
    _attr, _gain = None, 0.
    
    H_df = entropy(df, attr_tget)

    for attr, values in attrs_values.items():
        H_df_attr = 0.

        for value in values:
            df_attr_value = df[(df[attr] == value)]
            p_df_attr_value = df_attr_value.shape[0] / df.shape[0]
            H_df_attr += entropy(df_attr_value, attr_tget) * p_df_attr_value

        if H_df - H_df_attr > _gain: _attr, _gain = attr, (H_df - H_df_attr)

    return _attr, _gain

def evaluate(X, tree):
    y = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        node = tree
        while node.label not in [0, 1]:
            node = node.children[X[node.label].iloc[i]]
        y[i] = node.label
    
    return y