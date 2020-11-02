from spacy.lang.vi import Vietnamese
from pyvi.ViPosTagger import postagging
from collections import Counter
import re
import string


def clean_data(text):
    nlp = Vietnamese()
    text = text.lower().strip() #remove space and lower text
    text_remove_num = re.sub(r'\d+', ' ', text) #remove number
    text_remove_punct = text_remove_num.translate(str.maketrans("", "",string.punctuation)) #remove punct
    tokens = nlp(text_remove_punct)
    newtext = ""
    for token in tokens:
        if not token.is_stop:
            if postagging(str(token))[1] == ['N']:
                newtext = newtext + " " + str(token)   #select common noun from text
    word_keys = Counter(newtext.split(" ")) #Filter keys from text
    word_texts = ""
    for word in list(word_keys):
        word_texts = word_texts + " " + word
    return word_texts


def form_output(input):
    text_outputs = ['beauty', 'camera', 'home electric', 'sports and outsides', 'vehicles']
    return text_outputs[int(input)]




