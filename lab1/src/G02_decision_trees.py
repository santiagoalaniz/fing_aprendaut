# En este archivo de python se encuentra nuestra implementacion del algoritmo ID3, su diseño
# esta inspirado en los classifiers de sklearn, por ejemplo en el nombramiento
# de los metodos (fit, predict, score), y en la forma de pasar los parametros.
# Motivacion: Parecernos lo mas posible a los clasificadores de sklearn, para que 
# la comparacion sea sencilla.

import src.ID3_utils as utils
from sklearn.metrics import accuracy_score

class ID3Classifier():
    def __init__(self, attrs_values= {}, min_samples_split= 2, min_split_gain= 0.):
        self.min_samples_split = min_samples_split
        self.min_split_gain = min_split_gain
        self.tree = None
        self.attrs_values = attrs_values
        self.attrs = list(attrs_values.keys())

    def fit(self, X, y):
        X[y.name] = y
        self.tree = self.__id3(X, y.name, self.attrs)
    
    def predict(self, X):
        return utils.evaluate(X, self.tree)
    
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    # private

    ## Mitchell, p. 68
    def __id3(self, exs, attr_tget, attrs):
        same_value_attr_tget = exs[attr_tget].nunique() == 1
        attrs_empty = len(attrs) == 0
        
        if same_value_attr_tget or attrs_empty: return utils.node(exs[attr_tget].mode()[0])

        _attrs_values = { k: self.attrs_values[k] for k in attrs }
        best_attr, gain = utils.max_gain_attr(exs, attr_tget, _attrs_values)
        
        if gain <= self.min_split_gain: return utils.node(exs[attr_tget].mode()[0])
        
        node = utils.node(best_attr, gain)
        best_attr_values = self.attrs_values[best_attr]
        
        for attr_val in best_attr_values:
            exs_i = exs[exs[best_attr] == attr_val]

            if exs_i.shape[0] <= self.min_samples_split: 
              node.children[attr_val] = utils.node(exs[attr_tget].mode()[0])
            else:
              attrs_i = [attr for attr in attrs if attr != best_attr]
              node.children[attr_val] = self.__id3(exs_i, attr_tget, attrs_i)
        
        return node
        