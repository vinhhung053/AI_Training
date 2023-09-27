from gensim.models import KeyedVectors
import numpy as np


class Word2Vec:
    def __init__(self, data_file_path):
        # Load the Word2Vec model from the specified path
        model_path = data_file_path + 'baomoi.model.bin'
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        # Chuyển dấu _ thành dấu space trong từ điển
        self.model.index_to_key = [word.replace('_', ' ') for word in self.model.index_to_key]
        self.model.key_to_index = {word: idx for idx, word in enumerate(self.model.index_to_key)}

    def get_words(self):
        # Get the list of words in the model
        return self.model.index_to_key

    def get_size(self):
        # Get the size of words in the model
        return len(self.model.key_to_index)

    def get_word_vector(self, word):
        # Get the vector representation for a specific word
        return self.model[word]

    def find_similar_words(self, word, topn=5):
        # Find the most similar words to a given word
        return self.model.most_similar(word, topn=topn)

    def add_words_to_word2vec(self, x):
        words_to_add = []
        for sentences in x:
            for word in sentences:
                if word not in self.get_words() and word not in words_to_add:
                    words_to_add.append(word)

        vector_size = 400
        new_vector = np.random.rand(vector_size)
        vectors_to_add = [new_vector] * len(words_to_add)
        self.model.add_vectors(words_to_add, vectors_to_add)
        return self
