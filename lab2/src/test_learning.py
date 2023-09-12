import random
import numpy as np

def train_and_evaluate(clf, _N, _devel, ITERATIONS, Z):
    results = []
    for i in range(ITERATIONS):
        success = 0
        unpredicted_sentences = []
        for sentence in random.sample(_devel, _N): 
            aux = sentence[0:len(sentence)-1]
            suggested_word = clf.predict(aux)
            if (sentence[-1] == suggested_word): 
                success+=1
            else:
                unpredicted_sentences.append(' '.join(sentence))        
            
        for s in unpredicted_sentences:
            clf.update(s)

        _P = success/_N
        delta = Z*np.sqrt(_P*(1-_P)/_N)

        results.append((clf.N, i, _P-delta, _P+delta))
    return results
