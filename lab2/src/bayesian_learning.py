from naive_bayes_utils import *
from preprocessing import G02Preprocessor

from numpy import log
import pdb

class G02NaiveBayesClassifier:
    def __init__(self, data, N= 1, M=1):
        self.preprocessor = G02Preprocessor()
        self.N = N
        self.M = M
        self.V, self.F_h, self.F_hD = build(self.preprocessor.apply(data), N)
        
    
    def predict(self, sentence):
        argmax_ = {}

        for h in self.F_h.keys():
            p = log(p_h(h, self.V, self.F_h, self.M))
            for word in sentence:
                p += log(p_hD(word, h, self.V, self.F_h, self.F_hD, self.M))

            argmax_[h] = p
        
        return max(argmax_, key=argmax_.get)
    
    def update(self, new_sentence):
        self.F_h.update(new_sentence)

        for i in range(self.N, len(new_sentence)):
            current_word = new_sentence[i]
            previous_words = tuple(new_sentence[i-self.N:i])

            for previous_word in previous_words:
                self.F_hD[current_word].update([previous_word])
        
        self.V = sum(self.F_h.values())
        
