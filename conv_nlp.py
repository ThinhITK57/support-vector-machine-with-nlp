from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SeparableConv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models
from keras.regularizers import l2

import os
import random
import re
import string
from spacy.lang.vi import Vietnamese
from collections import Counter

from pyvi.ViPosTagger import postagging
import tensorflow as tf
import numpy as np

from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text


MAX_SEQUENCE_LENGTH = 500
TOP_K = 20000


def clean_data(text):
    nlp = Vietnamese()
    text = text.lower().strip()
    text_remove_num = re.sub(r'\d+', ' ', text)
    text_remove_punct = text_remove_num.translate(str.maketrans("", "",string.punctuation))
    tokens = nlp(text_remove_punct)
    newtext = ""
    for token in tokens:
        if not token.is_stop:
            if postagging(str(token))[1] == ['N']:
                newtext = newtext + " " + str(token)
    word_keys = Counter(newtext.split(" "))
    word_texts = ""
    for word in list(word_keys):
        word_texts = word_texts + " " + word
    return word_texts


def load_dataset(data_path, seed=123):
    train_text = []
    train_label = []
    for category in ['b', 'c', 'h', 's', 'v']:
        train_path = os.path.join(data_path, 'train', category)
        for file_name in sorted(os.listdir(train_path)):
            if file_name.endswith('.txt'):
                with open(os.path.join(train_path, file_name), "r", encoding="utf-8") as f:
                    train_text_keys = clean_data(f.read())
                    train_text.append(train_text_keys)
                train_label.append(category)
    test_text = []
    test_label = []

    for category in ['b', 'c', 'h', 's', 'v']:
        test_path = os.path.join(data_path, 'test', category)
        for file_name in sorted(os.listdir(test_path)):
            if file_name.endswith('.txt'):
                with open(os.path.join(test_path, file_name), "r", encoding="utf-8") as f:
                    test_text_keys = clean_data(f.read())
                    test_text.append(test_text_keys)
                test_label.append(category)

    random.seed(seed)
    random.shuffle(train_text)
    random.seed(seed)
    random.shuffle(train_label)

    return((train_text, np.array(train_label)),
            (test_text, np.array(test_label)))


def _get_last_layer_units_and_activation(num_classes):
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def sequence_vectorize(train_texts, val_texts):
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index


def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    if use_pretrained_embedding:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable))
    else:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0]))

    for _ in range(blocks - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, kernel_regularizer=l2(0.01)))
    model.add(Activation('linear'))
    model.add(Flatten())
    return model


def train_sequence_model(data,
                         learning_rate=1e-3,
                         epochs=1000,
                         batch_size=128,
                         blocks=2,
                         filters=64,
                         dropout_rate=0.2,
                         embedding_dim=200,
                         kernel_size=3,
                         pool_size=3):
    (train_texts, train_labels), (val_texts, val_labels) = data
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_labels)
    val_labels = encoder.transform(val_labels)
    num_classes = 5
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(unexpected_labels=unexpected_labels))

    x_train, x_val, word_index = sequence_vectorize(train_texts, val_texts)

    num_features = min(len(word_index) + 1, TOP_K)

    model = sepcnn_model(blocks=blocks,
                         filters=filters,
                         kernel_size=kernel_size,
                         embedding_dim=embedding_dim,
                         dropout_rate=dropout_rate,
                         pool_size=pool_size,
                         input_shape=x_train.shape[1:],
                         num_classes=num_classes,
                         num_features=num_features)
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='hinge', metrics=['acc'])
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('rotten_tomatoes_sepcnn_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == '__main__':

    data = load_dataset("data")
    train_sequence_model(data)