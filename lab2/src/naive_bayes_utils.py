from collections import Counter, defaultdict

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

    V = sum(F_h.values())

    return V, F_h, F_hD

def p_h(h, V, F_h, m=1):
    p = 1/ len(F_h)
    F_h_value = F_h[h]
    
    return (F_h_value + m * p) / (len(F_h) + m)

def p_hD(d, h, V, F_h, F_hD, m=1):
    p = 1/ len(F_h)
    F_hD_given_h = F_hD[h][d] if d in F_hD[h] else 0
    F_h_value = F_h[h]

    return (F_hD_given_h + m * p) / (F_h_value + m)
