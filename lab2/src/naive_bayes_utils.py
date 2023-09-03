from collections import Counter, defaultdict

def build(data, N):
    F_h = Counter()
    F_hD = defaultdict(Counter)

    for sentence in data:
        F_h.update(sentence)

        for i in range(N, len(sentence)):
            current_word = sentence[i]
            previous_words = tuple(sentence[i-N:i])

            for previous_word in previous_words:
                F_hD[current_word].update([previous_word])

    V = sum(F_h.values())

    return V, F_h, F_hD

def p_h(h, V, F_h, m=1):
    p = 1 / V
    F_h_value = F_h[h] if h in F_h else 0
    
    return (F_h_value + m * p) / (V + m)

def p_hD(d, h, V, F_h, F_hD, m=1):
    p = 1 / V
    F_hD_given_h = F_hD[h][d] if h in F_hD and d in F_hD[h] else 0
    F_h_value = F_h[h] if h in F_h else 0

    return (F_hD_given_h + m * p) / (F_h_value + m)
