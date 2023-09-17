from gensim.models import KeyedVectors

# Đường dẫn đến tệp mô hình Word2Vec đã tải về
model_path = '/home/lap13385/Projects/Week3_4_NLP/RNN_Classification/baomoi.model.bin'

# Tải mô hình Word2Vec
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
if 'dây_cáp' in model.index_to_key:
    print('hung')
model.index_to_key = [word.replace('_', ' ') for word in model.index_to_key]
model.key_to_index = [word.replace('_', ' ') for word in model.key_to_index]
print(len(model.key_to_index))
# print(model.key_to_index)
if 'dây cáp' in model.index_to_key:
    print('hung')

# print(model['dây_cáp'])

# for word in model.index_to_key:
#     print(word)
# model.add_vector('dây cáp',[1]*400)
# word_vector = model['dây cáp']
# print(word_vector)
# print(len(model.key_to_index))

# Tìm các từ tương tự
# similar_words = model.most_similar('tiki training', topn=5)
# print(similar_words)