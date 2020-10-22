import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.layers as layers
from sklearn.model_selection import train_test_split
from time import time
from os.path import abspath, join, pardir, dirname, lexists
from src.data_cleaning import *
from tensorflow.keras import Model
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print("gpus found- {}".format(gpus))
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
file_path = dirname(abspath(__file__))
project_path = abspath(join(file_path, pardir))

VOCAB_SIZE = 10000
OOV_TOKEN = '<OOV>'
TRUNCATING = 'pre'
PADDING = 'pre'
MAX_SENTENCE_LENGTH = 120
EMBEDDING_DIM = 100

train_data = pd.read_csv(join(project_path, 'input/train.csv'), index_col='id')
train_data = train_data.append({'qid1': 303951, 'qid2': 174363, 'question1': 'How can I create an Android app?',
                                'question2': 'How can I develop android app?', 'is_duplicate': 1}, ignore_index=True)
train_data = train_data.dropna(axis=0)
# train_data.question1 = train_data.question1.apply(lambda x: remove_stopwords(remove_apostrophe(x)))
# train_data.question2 = train_data.question2.apply(lambda x: remove_stopwords(remove_apostrophe(x)))
train_data.question1 = train_data.question1.apply(lambda x: remove_stopwords(remove_apostrophe(x)))
train_data.question2 = train_data.question2.apply(lambda x: remove_stopwords(remove_apostrophe(x)))
X_train, X_valid, y_train, y_valid = train_test_split(train_data[['question1', 'question2']],
                                                      train_data['is_duplicate'])
print("training samples: {}, validation samples: {}".format(X_train.shape, X_valid.shape))

start_time = time()


tokenizer = Tokenizer(oov_token=OOV_TOKEN, num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(X_train.question1.append(X_train.question2))
# word_index_dict = list(tokenizer.word_index.keys())[:VOCAB_SIZE]


def convert_to_seqs(x):
    seqs = tokenizer.texts_to_sequences(x)
    return pad_sequences(seqs, maxlen=MAX_SENTENCE_LENGTH, truncating=TRUNCATING, padding=PADDING)


train_q1_sents = convert_to_seqs(X_train.question1.copy())
train_q2_sents = convert_to_seqs(X_train.question2.copy())
valid_q1_sents = convert_to_seqs(X_valid.question1.copy())
valid_q2_sents = convert_to_seqs(X_valid.question2.copy())
print("training input: {}, validation input: {}".format(train_q1_sents.shape, valid_q1_sents.shape))

embedding = 2 * np.random.randn(VOCAB_SIZE + 1, EMBEDDING_DIM) - 1
vocab_word_index = [key for key, value in tokenizer.word_index.items() if value < VOCAB_SIZE + 1]
with open('../../../glove.6B/glove.6B.{}d.txt'.format(EMBEDDING_DIM), 'r') as f:
    data = f.readlines()
    embedding_dict = {}
    count = 0
    missed_words = []
    for vec in data:
        val = vec.split(' ')
        embedding_dict[val[0]] = np.asarray(val[1:])
    for ind, word in enumerate(vocab_word_index):
        if word in embedding_dict.keys():
            embedding[ind, :] = embedding_dict[word]
        else:
            missed_words.append(word)
    print(len(embedding_dict), len(missed_words))
    print(missed_words)
if not lexists("model/saved_model.pb"):

    q1_input = layers.Input((MAX_SENTENCE_LENGTH,), sparse=False)
    q2_input = layers.Input((MAX_SENTENCE_LENGTH,), sparse=False)

    embedding_layer = layers.Embedding(input_dim=VOCAB_SIZE+1, output_dim=EMBEDDING_DIM, input_length=MAX_SENTENCE_LENGTH,
                                       embeddings_initializer=tf.keras.initializers.Constant(embedding),
                                       trainable=False)
    q1_embedding = embedding_layer(q1_input)
    q2_embedding = embedding_layer(q2_input)

    q1_recurrent = layers.Bidirectional(layer=layers.GRU(128))(q1_embedding)
    q2_recurrent = layers.Bidirectional(layer=layers.GRU(128))(q2_embedding)

    q1_dense = layers.Dense(64, activation='relu')(q1_recurrent)
    q2_dense = layers.Dense(64, activation='relu')(q2_recurrent)

    combined_layer = layers.concatenate([q1_dense, q2_dense])
    combined_dense_layer = layers.Dense(32, activation='relu')(combined_layer)
    output_layer = layers.Dense(1, activation='sigmoid')(combined_dense_layer)

    complete_model = Model(inputs=[q1_input, q2_input], outputs=[output_layer])
    print(complete_model.summary())
    complete_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    input_array = np.concatenate([train_q1_sents, train_q2_sents], axis=1)
    print(input_array.shape)
    complete_model.fit([train_q1_sents, train_q2_sents], y_train, batch_size=512, epochs=10,
                       validation_data=([valid_q1_sents, valid_q2_sents], y_valid))

    complete_model.save("model")
time1 = time()
print(time1 - start_time)
model = tf.keras.models.load_model("model")
print(join(project_path, 'input/test.csv'))
test_data = pd.read_csv(join(project_path, 'input/test.csv'))
print(test_data.shape, test_data.isnull().sum())

test_data.question2 = test_data.question2.fillna('How I what can learn android app development?')
test_data.question1 = test_data.question1.fillna('How app development?')
print(test_data.loc[[1046690,1461432,379205,817520,943911,1270024], :])

print(test_data.shape, test_data.columns)
test_q1_sents = convert_to_seqs(test_data.question1)
test_q2_sents = convert_to_seqs(test_data.question2)

predictions = model.predict([test_q1_sents, test_q2_sents])
print(predictions[:10, 0])
print(test_data.test_id)
submission = pd.DataFrame({'is_duplicate': predictions[:, 0], 'test_id': test_data.test_id})
submission.to_csv('bi_gru_submission.csv', index=False)
print(submission.head())
print("load and predict time: {}", (time() - time1))
