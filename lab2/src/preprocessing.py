from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode

class G02Preprocessor:
    def __init__(self):
        with open('assets/es_words.txt', 'r') as f:
            words = f.read().split('\n') + stopwords.words('spanish')

        self.V_SPA = set([unidecode(word.lower()) for word in words])

    def apply(self, data, data_test=True):
        preprocessed_data = []

        for message in data:
            words = word_tokenize(message, language='spanish')
            words = [unidecode(word.lower()) for word in words]
            words = [word for word in words if word.isalpha()]
            if not words: continue
            if data_test:
                words = [word for word in words if word in self.V_SPA]
            if not words: continue

            preprocessed_data.append(words)

        return preprocessed_data
