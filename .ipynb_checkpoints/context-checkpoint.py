import pandas as pd
import json
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')

def count_words(text):
    count_dict = {}
    for word in text:
        if word not in count_dict:
            count_dict[word] = 1
        else:
            count_dict[word] += 1
    print(count_dict)
    return count_dict

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    contractions = json.load(open('contractions.json','r'))
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions.keys():
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = [lemmatizer.lemmatize(x) for x in text.split()]

    return text



def main(txt):
    text = clean_text(txt)
    wc = count_words(text)
    return wc
