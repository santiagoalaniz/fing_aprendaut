# En este archivo de python se encuentra nuestra implementacion del algoritmo ID3, su diseño
# esta inspirado en los classifiers de sklearn, por ejemplo en el nombramiento
# de los metodos (fit, predict, score), y en la forma de pasar los parametros.
# Motivacion: Parecernos lo mas posible a los clasificadores de sklearn, para que 
# la comparacion sea sencilla.

import src.ID3_utils as utils
from sklearn.metrics import accuracy_score

class ID3Classifier():
    def __init__(self, min_samples_split=0, min_split_gain=0., attrs_values={}):
        self.min_samples_split = min_samples_split
        self.min_split_gain = min_split_gain
        self.tree = None
        self.attrs_values = attrs_values

    def fit(self, X, y):
        X['Target'] = y
        self.tree = self.__id3(X, 'Target', X.iloc[:, :-1].columns)
    
    def predict(self, X):
        return utils.evaluate(X, self.tree)
    
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
    
    # private

    ## Mitchell, p. 68
    def __id3(self, exs, attr_tget, attrs):
        if exs[attr_tget].nunique() == 1: return utils.node(1)
        if exs[attr_tget].nunique() == 0: return utils.node(0)
        if len(attrs) == 0: return utils.node(exs[attr_tget].mode()[0])

        best_attr, gain = utils.max_gain_attr(exs, attr_tget, attrs)
        
        if gain <= self.min_split_gain: return utils.node(exs[attr_tget].mode()[0])
        
        node = utils.node(best_attr, gain)
        best_attr_values = self.attrs_values[best_attr]
        
        for attr_val in best_attr_values:
            exs_i = exs[exs[best_attr] == attr_val]

            if exs_i.shape[0] <= self.min_samples_split: 
              node.children[attr_val] = utils.node(exs[attr_tget].mode()[0])
            else: 
              node.children[attr_val] = self.__id3(exs_i, attr_tget, attrs.drop(best_attr))
        
        return node
        