# En este archivo de python se encuentra nuestra implementacion del algoritmo ID3, su diseño
# esta inspirado en el modulo sklearn.tree.DecisionTreeClassifier, por ejemplo en el nombramiento
# de los metodos (fit, predict, score), y en la forma de pasar los parametros.
# Motivacion: Parecernos lo mas posible a los clasificadores de sklearn, para que sea mas
# facil comparar con los resultados.

import pandas as pd
import src.ID3_utils as utils

class G02ID3Classifier():
    def __init__(self, min_samples_split=0, min_split_gain=0):
        self.min_samples_split = min_samples_split
        self.min_split_gain = min_split_gain

    def fit(self, X, y):
        self.attributes = X.columns.values
        self.attributes_values = {attr: X[attr].unique() for attr in self.attributes}
        X['Target'] = y
    
    def predict(self, X):
        return 1
    
    def score(self, X, y):
        return 1
    
    # private methods

    def __id3(self):
        return 1
        

def main():
    return 1

if __name__ == "__main__":
    main()