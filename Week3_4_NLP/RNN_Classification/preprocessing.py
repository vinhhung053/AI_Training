import re
import string
from underthesea import word_tokenize

class Prerpocessing:
    def __init__(self, data_file_path):
        with open(data_file_path + "vietnamese-stopwords.txt", 'r') as f:
            self.stop_words = f.read().split('\n')

    def remove_punctuation(self, text):
        text = re.sub(f'[{string.punctuation}\d\n]', '', text)
        return text

    def tokenize(self, text):
        tokens = word_tokenize(text.lower())
        return tokens

    def remove_stopwords(self, tokens):
        tokens_new = [w for w in tokens if w not in self.stop_words or w != '']
        return tokens_new
    def preprocess_text(self, text):
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        return tokens