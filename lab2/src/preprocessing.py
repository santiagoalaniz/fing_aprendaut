import nltk
from nltk.corpus import stopwords, cess_esp
from nltk.tokenize import word_tokenize
from unidecode import unidecode

nltk.download('stopwords', quiet=True)
nltk.download('cess_esp', quiet=True)

class G02Preprocessor:
    def __init__(self):
        self.spanish_words = set([unidecode(word.lower()) for word in cess_esp.words()])
        
    
    def apply(self, data, data_test=True):
        preprocessed_data = []

        for message in data:
            words = word_tokenize(message, language='spanish')
            words = [unidecode(word.lower()) for word in words]
            words = [word for word in words if word.isalpha()]
            if not words: continue
            words = [word for word in words if not word in stopwords.words('spanish')]
            if not words: continue
            words = [word for word in words if word in self.spanish_words]
            if not words: continue

            preprocessed_data.append(words)

        return preprocessed_data