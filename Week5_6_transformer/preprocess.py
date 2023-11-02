import re
import string
from underthesea import word_tokenize
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class Prerpocessing:

    def tokenize(self, text):
        tokens = text.split()
        return tokens

    def remove_punctuation(self, text):
        text = re.sub(f'[{string.punctuation}\d\n]', '', text)
        text = re.sub(f'—', '', text)
        return text

    def lower_text(selft,text):
        return text.lower()

    def preprocess_text(self, text):
        text = self.remove_punctuation(text)
        text = self.lower_text(text)
        tokens = self.tokenize(text)
        # print(tokens)
        return tokens
