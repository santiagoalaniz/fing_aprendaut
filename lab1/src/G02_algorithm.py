import pandas as pd
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from src.nodo import Nodo

''' Algoritmo básico (notas de decisión)
Crear una raíz
• Si todos los ej. tienen el mismo valor → etiquetar con ese valor
• Si no me quedan atributos → etiquetar con el valor más común
• En caso contrario:
‣ La raíz pregunta por A, atributo que mejor clasifica los ejemplos
‣ Para cada valor vi de A
๏ Genero una rama
๏ Ejemplosvi={ejemplos en los cuales A=vi }
๏ Si Ejemplosvi es vacío → etiquetar con el valor más probable
๏ En caso contrario → ID3(Ejemplosvi, Atributos -{A})
'''


class CustomID3Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, MIN_SAMPLES_SPLIT=0, MIN_SPLIT_GAIN=0):
        self.MIN_SAMPLES_SPLIT = MIN_SAMPLES_SPLIT
        self.MIN_SPLIT_GAIN = MIN_SPLIT_GAIN

    ###########################################################################
    # FIT
    ###########################################################################
    def fit(self, X: pd.DataFrame, y):
        self.root = self.__fit(X, y, X.columns.values, "root", 0)

    def __fit(self, X: pd.DataFrame, y, attributes, attr, value):
        # Caso base 1: todos los datos tienen la misma clase
        if y.nunique() == 1:
            return Nodo(attr, value, tipo=Nodo.hoja, resultado=y.unique()[0], n_casos=X.shape[0])
        # Caso base 2: no hay mas atributos
        elif len(attributes) == 0:
            return Nodo(attr, value, tipo=Nodo.hoja, resultado=y.mode()[0], n_casos=X.shape[0])

        # Caso recursivo
        else:
            # Seleccionar el mejor atributo
            best_attribute, gain, unique_values = self.__get_best_attribute(
                X, y, attributes)
            
            # if gain < MIN_SPLIT_GAIN:
            #     print_value(value)
            #     return Nodo(attr, value, tipo=Nodo.hoja, resultado= y.mode()[0])

            # if len(unique_values) < MIN_SAMPLES_SPLIT:
            #     print_value(value)
            #     return Nodo(attr, value, tipo=Nodo.hoja, resultado= y.mode()[0])

            # Eliminar el atributo de la lista de atributos
            attributes = attributes[attributes != best_attribute]
            # Crear un nodo hijo por cada valor del atributo
            children = []
            for child_value in unique_values:
                subdata = X[X[best_attribute] == child_value]
                y_sub = y[X[best_attribute] == child_value]
                child = self.__fit(subdata, y_sub, attributes,
                                   best_attribute, child_value)

                children.append(child)

            return Nodo(attr, value, tipo=Nodo.rama, hijos=children, n_casos=X.shape[0])

    def __get_best_attribute(self, X, y, attributes):
        best_attribute = None
        best_gain = 0
        best_unique_values = None
        for attribute in attributes:
            gain, unique_values = self.__get_gain(X, y, attribute)
            if gain >= best_gain:
                best_attribute = attribute
                best_gain = gain
                best_unique_values = unique_values
        return best_attribute, best_gain, best_unique_values

    def __get_gain(self, X, y, attribute):
        gain = self.__get_entropy(y)
        unique_values = X[attribute].unique()
        for value in unique_values:
            subdata = y[X[attribute] == value]
            gain -= (subdata.shape[0] / X.shape[0]) * \
                self.__get_entropy(subdata)
        return gain, unique_values

    def __get_entropy(self, y):
        entropy = 0
        for value in y.unique():
            p = y[y == value].shape[0] / y.shape[0]
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    ###########################################################################
    # PREDICT
    ###########################################################################
    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            y_pred.append(self.__get_result(row, self.root))
        return y_pred

    def __get_next_node(self, row, nodo):
        max_n_casos = 0
        max_n_casos_hijo = None

        for hijo in nodo.hijos:
            if hijo.n_casos > max_n_casos:
                max_n_casos = hijo.n_casos
                max_n_casos_hijo = hijo
            if hijo.valor == row[hijo.attr]:
                return hijo

        return max_n_casos_hijo

    def __get_result(self, row, nodo):
        if nodo.tipo == Nodo.hoja:
            return nodo.resultado
        else:
            return self.__get_result(row, self.__get_next_node(row, nodo))
