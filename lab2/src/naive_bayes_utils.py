from collections import Counter, defaultdict
from numpy import log2

def build(data, N):
    F_h = Counter()
    F_hD = defaultdict(Counter)

    for sentence in data:
        F_h.update(sentence)

        for i in range(0, len(sentence)):
            current_word = sentence[i]
            previous_words = sentence[max(0, i - N):i]

            for previous_word in previous_words:
                F_hD[current_word].update([previous_word])

    N_OF_WORDS = sum(F_h.values())

    return N_OF_WORDS, F_h, F_hD

def p_h(h, N_OF_WORDS, F_h, data):
    fr = F_h[h] / N_OF_WORDS
    h_in_data = len([x for x in data if h in x])
    return fr * log2(len(data) / (h_in_data))

def p_hD(d, h, V_SPA, F_h, F_hD, m=1):
    p = 1/ len(V_SPA)
    F_hD_given_h = F_hD[h].get(d, 0)
    F_h_value = F_h[h]

    return (F_hD_given_h + m * p) / (F_h_value + m)
