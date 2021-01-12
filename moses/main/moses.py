#! /usr/bin/env python
#
# Douglas Amante
# Date: 06.01.2021
"""
Example using  MOSES evolutionary program
learning system for Fakenews classification

Another detail, for complete documentation on how to pass additional parameters to
MOSES, refer to the documentation at the following way:
moses_to_aa/moses/moses/cython/opencog$/pymoses.py

MOSES Details:
http://wiki.opencog.org/w/Meta-Optimizing_Semantic_Evolutionary_Search
"""

__author__ = 'Cosmo Harrigan and Update by Douglas Amante'

from opencog.pymoses import *

moses = moses()

data_train = pd.read_csv('/moses_to_aa/moses/moses/moses/main/dataset\snopes.csv')
data_train.text[1]
# Input Data preprocessing

#data_train['claim'] = data_train['claim'].replace('FAKE',1)
#data_train['claim'] = data_train['claim'].replace('REAL',0)

texts = []
labels = []

#Using Pre-trained word embeddings
GLOVE_DIR = "data" 
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, '/moses_to_aa/moses/moses/moses/main/dataset/glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    #print(values[1:])
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

print "\nTraining data:\n\n{0}".format(data_train)


model = moses.eval(sequence_input, preds)

output = moses.run(input=data_train, python=True)

print moses.run(input=data_train, python=False)[0].program
