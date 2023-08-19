
import math 

def get_best_attribute(data, attributes, target):
    best_attribute = None
    best_gain = 0
    best_unique_values = None
    for attribute in attributes:
        gain, unique_values = get_gain(data, attribute, target)
        if gain > best_gain:
            best_attribute = attribute
            best_gain = gain
            best_unique_values = unique_values
    return best_attribute, best_gain, best_unique_values

def get_gain(data, attribute, target):
    gain = get_entropy(data, target)
    unique_values = data[attribute].unique()
    for value in unique_values:
        subdata = data[data[attribute] == value]
        gain -= (subdata.shape[0] / data.shape[0]) * get_entropy(subdata, target)
    return gain, unique_values

def get_entropy(data, target):
    entropy = 0
    for value in data[target].unique():
        p = data[data[target] == value].shape[0] / data.shape[0]
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy