# keras embedding mask_zeroの説明
# https://qiita.com/hrappuccino/items/f66abebe60f8ea7826d5
# imdbのtips
# http://uchidama.hatenablog.com/entry/2018/02/01/063200

from __future__ import print_function

from keras.preprocessing import sequence
from keras import Model
from keras.layers import Input, Dense
from keras.layers import Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
import tensorflow as tf

# GPU setting
config = tf.ConfigProto(
            gpu_options = tf.GPUOptions(
                visible_device_list="0", # specify GPU number
                allow_growth=True)
        )

# for ELMo
import tensorflow_hub as hub
from keras import backend as K
from keras.layers import Concatenate, Lambda
import numpy as np
sess = tf.Session(config=config)
K.set_session(sess)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

# hyper parameters
max_features = 20000
maxlen = 80
batch_size = 100
use_elmo = True

# 生データに戻す
def edit_data(x_train, x_test):
    x_train_raw = []
    x_test_raw = []
    INDEX_FROM = 3

    # Make Word to ID dictionary
    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
    word_to_id["[PAD]"] = 0
    word_to_id["[🏃]"] = 1 # START
    word_to_id["[❓]"] = 2 # UNKNOWN

    # Make ID to Word dictionary
    id_to_word = {value:key for key,value in word_to_id.items()}

    # Make raw caption data
    for train_data in x_train:
        word_list = []
        for id in train_data:
            word_list.append(id_to_word[id])
        x_train_raw.append(word_list)

    for test_data in x_test:
        word_list = []
        for id in test_data:
            word_list.append(id_to_word[id])
        x_test_raw.append(word_list)
    
    return np.asarray(x_train_raw), np.asarray(x_test_raw)

def load_data():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    x_train_raw, x_test_raw = edit_data(x_train, x_test)
    print('x_train_raw shape:', x_train_raw.shape)
    print('x_test_raw shape:', x_test_raw.shape)

    return x_train, x_train_raw, x_test, x_test_raw, y_train, y_test

def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[maxlen])},
                      signature="tokens",
                      as_dict=True)["elmo"]

def build_model():
    # GloVeのみで初期化したモデルも用意したい
    if use_elmo is False:
        print('Build model without ELMo...')
        _input = Input(shape=(maxlen, ))
        _embed = Embedding(input_dim=max_features, output_dim=300)(_input)
        _lstm = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))(_embed)
        _output = Dense(1, activation='sigmoid')(_lstm)

        return Model(_input, _output)

    elif use_elmo is True:
        print('Build model with ELMo...')
        _input = Input(shape=(maxlen, ))
        _raw_input = Input(shape=(maxlen, ), dtype=tf.string)
        _embed = Embedding(input_dim=max_features, output_dim=300, mask_zero=True)(_input)
        _elmo_embed = Lambda(ElmoEmbedding, output_shape=(maxlen, 1024))(_raw_input)
        _embed = Concatenate(axis=-1)([_embed, _elmo_embed])
        _lstm = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))(_embed)
        _output = Dense(1, activation='sigmoid')(_lstm)

        return Model([_input, _raw_input], _output)

def train(model, x_train, x_train_raw, x_test, x_test_raw, y_train, y_test):
    model.summary()
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Train...')
    callbacks = [EarlyStopping(patience=0, verbose=1)]

    if use_elmo is True:
        model.fit([x_train, x_train_raw], y_train,
                batch_size=batch_size,
                epochs=15,
                validation_data=([x_test, x_test_raw], y_test),
                callbacks=callbacks)
        score, acc = model.evaluate([x_test, x_test_raw], y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
    elif use_elmo is False:
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=15,
                validation_data=(x_test, y_test),
                callbacks=callbacks)
        score, acc = model.evaluate(x_test, y_test,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

if __name__ == '__main__':
    x_train, x_train_raw, x_test, x_test_raw, y_train, y_test = load_data()
    model = build_model()
    train(model, x_train, x_train_raw, x_test, x_test_raw, y_train, y_test)