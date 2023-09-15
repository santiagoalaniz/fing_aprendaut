from src.naive_bayes_utils import *
from src.preprocessing import G02Preprocessor
from numpy import log2

class G02NaiveBayesClassifier:
    def __init__(self, data, N= 1, M=1):
        self.N = N
        self.M = M
        self.preprocessor = G02Preprocessor()
        self.data = self.preprocessor.apply(data)
        self.V_SPA = self.preprocessor.V_SPA
        self.N_OF_WORDS, self.F_h, self.F_hD = build(self.data, N)


    def predict(self, sentence):
        argmax_ = {}
        sentence = sentence[-self.N:]

        for h in self.F_h.keys():
            p = log2(p_h(h, self.N_OF_WORDS, self.F_h, self.data))

            for word in sentence:
                p += log2(p_hD(word, h, self.V_SPA, self.F_h, self.F_hD, self.M))

            argmax_[h] = p

        h_map = max(argmax_, key=argmax_.get)

        while h_map in sentence:
            argmax_.pop(h_map)
            h_map = max(argmax_, key=argmax_.get)

        return h_map

    def update(self, new_sentence):
        new_sentence = self.preprocessor.apply([new_sentence])
        if not new_sentence: return new_sentence

        preprocessed_sentence = new_sentence[0]
        self.F_h.update(preprocessed_sentence)

        for i in range(0, len(preprocessed_sentence)):
            current_word = preprocessed_sentence[i]
            previous_words = preprocessed_sentence[max(0, i - self.N):i]

            for previous_word in previous_words:
                self.F_hD[current_word].update([previous_word])

        self.data += preprocessed_sentence
        self.N_OF_WORDS += len(preprocessed_sentence)

        return new_sentence
