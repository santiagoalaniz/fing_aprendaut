import nltk
from unidecode import unidecode

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

DICTIONARY_SPA_PATH = 'assets/diccionario.txt'

class G02Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('spanish'))

        with open(DICTIONARY_SPA_PATH, 'r', encoding='utf-8') as file:
            dictionary = file.read().split('\n')

        self.spanish_words = set([unidecode(word.lower()) for word in dictionary])
    
    def apply(self, data):
        preprocessed_data = []

        for message in data:
            words = nltk.word_tokenize(message, language='spanish')
            words = [unidecode(word.lower()) for word in words]
            words = [word for word in words if word.isalpha()]
            words = [word for word in words if not word in self.stop_words]
            words = [word for word in words if len(word) > 1]
            words = [word for word in words if word in self.spanish_words]
            
            if len(words) == 0: continue

            preprocessed_data.append(words)

        return preprocessed_data