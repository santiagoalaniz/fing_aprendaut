import nltk
from unidecode import unidecode

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('cess_esp', quiet=True)

from nltk.corpus import stopwords, cess_esp

class G02Preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('spanish'))
        self.spanish_words = set([unidecode(word.lower()) for word in cess_esp.words()])
    
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