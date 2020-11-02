import os
import random
import numpy as np
import re
import string
from spacy.lang.vi import Vietnamese
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import f_classif
from pyvi.ViPosTagger import postagging
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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


def best_ks(tup):
    X_train, y_train, X_test, y_test, label, k = tup

    # best features for label
    if k:
        kb = SelectKBest(mutual_info_classif, k=k)
        X_train_best = kb.fit_transform(X_train, y_train)
        X_test_best = kb.transform(X_test)
        return k, label, X_train_best, y_train, X_test_best, y_test
    else:
        return k, label, X_train, y_train, X_test, y_test


def ngram_vectorize(train_texts, train_labels, val_texts):
    vectorizer = TfidfVectorizer(min_df=3, encoding='utf-8', norm='l2', use_idf=True, decode_error='ignore')
    fit_vector = vectorizer.fit(train_texts)
    x_train = fit_vector.transform(train_texts)
    x_val = vectorizer.transform(val_texts)

    selector = SelectKBest(f_classif, k=min(20000, x_train.shape[1]))
    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype('float64')
    x_val = selector.transform(x_val).astype('float64')
    return x_train, x_val, vectorizer.vocabulary_


def _get_last_layer_units_and_activation(num_classes):
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation

from sklearn.preprocessing import LabelEncoder

"""def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))
    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model
"""


def train_ngram_model(data):
    (train_texts, train_labels),(val_texts, val_labels) = load_dataset(data)
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

    vectorize = TfidfVectorizer(min_df=3, norm='l2', use_idf=True, decode_error='ignore')
    fit_vector = vectorize.fit(train_texts)
    x_train = fit_vector.transform(train_texts)
    x_val = fit_vector.transform(val_texts)
    return fit_vector, x_train, x_val, train_labels, val_labels


fit_vector, x_train, x_val, train_labels, val_labels = train_ngram_model('data')
fit_filename = 'fit_vector.pkl'
pickle.dump(fit_vector, open(fit_filename, 'wb'))

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
SVM.fit(x_train,train_labels)

filename = 'model_svm.pkl'
pickle.dump(SVM, open(filename, 'wb'))

#prediction_svm = SVM.predict(x_val)
"""conf_mat = confusion_matrix(val_labels, prediction_svm)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=['beauty', 'camera', 'home', 'sports', 'vehicles'],
            yticklabels=['beauty', 'camera', 'home', 'sports', 'vehicles'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - SVM\n", size=16)
plt.savefig('svm.png')"""
#print("svm accuracy score -> ", accuracy_score(prediction_svm, val_labels)*100)











