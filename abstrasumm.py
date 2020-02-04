import nltk

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import en_core_web_sm
import heapq
import json
import matplotlib
import re
import spacy
import string
import tensorflow as tf

from collections import Counter
from collections import Counter
from nltk import ne_chunk
from nltk.chunk import conlltags2tree
from nltk.chunk import tree2conlltags
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pprint import pprint
from spacy import displacy
matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
import random
import re
import pandas as pd

from wordcloud import WordCloud

import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import datetime

SCOPES = ['https://www.googleapis.com/auth/calendar_entry']


f = open('convotext.txt', 'r').read().lower()

#     f = re.sub(r'\s+', ' ', f)
no_of_lines = len(open('convotext.txt', 'r').readlines())
stop_words = set(
    stopwords.words('english') +
    ['i', 'he', 'me', 'she', 'it', 'them', 'her', 'him'])

# print(stop_words)
# ! cd "/gdrive/My Drive/Colab_ML/Abstractive Summarizer" && wget "https://conceptnet.s3.amazonaws.com/downloads/2017/numberbatch/numberbatch-en-17.06.txt.gz"
# ! cd "/gdrive/My Drive/Colab_ML/Abstractive Summarizer/" && gunzip "/gdrive/My Drive/Colab_ML/Abstractive Summarizer/numberbatch-en-17.06.txt.gz"


def remove_punc(sent):
    #
    punctuations = '''!()-[]{};'"\,<>/?@#%^&*_~'''
    for x in sent:
        if x in punctuations:
            sent = sent.replace(x, "")
    return sent


def preprocess(sent):
    #
    sent = remove_punc(sent)
    sent = nltk.word_tokenize(sent, language='english')
    lemmatizer = WordNetLemmatizer()
    sent = [lemmatizer.lemmatize(x) for x in sent]
    sent = ' '.join(sent)
    filtered_sentence = [
        w for w in sent.split(' ') if not w.lower() in stop_words
    ]

    return ' '.join(filtered_sentence).lower()


def weighted_freq(sent):
    word_frequencies = {}
    for word in sent:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    return word_frequencies


def sent_score_calc(text, word_frequencies):
    sentence_list = nltk.sent_tokenize(text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                # if len(sent.split(' ')) < 10:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
    return sentence_scores


def extractive_summary(f, docu):
    max_freq = weighted_freq(docu)
    sent_scores = sent_score_calc(f, max_freq)
    no_of_lines = len(docu.split('.'))
    summary_sentences = heapq.nlargest(
        int(no_of_lines / 2), sent_scores, key=sent_scores.get)
    # summary_sentences =sorted(sent_scores, key=sent_scores.get, reverse=True)[:int(no_of_lines/2)]

    summary = ' '.join(summary_sentences)
    # print(summary)
    return summary


def return_context(docu):
    nlp = en_core_web_sm.load()
    doc = nlp(docu)
    fin_dic = {}
    for ent in doc.ents:
        fin_dic[ent.text] = ent.label_
    return json.dumps(fin_dic, sort_keys=True)


# pass a list with multiple conversations in this function. pls pass as a list pls pls
def trends(js):
    js = js.split('.')
    lis_trend = []
    for each in js:
        each = preprocess(each)
        lis_trend.extend(each.split(' '))

    dict_trend = Counter(lis_trend)
    dict_trend['.'] = 0

    wordcloud = WordCloud(
        width=500,
        height=500,
        background_color='white',
        stopwords=stop_words,
        min_font_size=7).generate(' '.join(list(set(lis_trend))))
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.show()
    plt.axis('off')
    plt.savefig('trends.png')

    return dict_trend.most_common(5)


def new_train_gen():
    # l = ['my no is 9003401119','is your phone 9341234441','your phone no is 8341934568','here is my no 8261348649','here is my no 6713401897']
    l = [
        "my email is ~msubhaditya@gmail.com",
        "is your email id ~rules@yahoo.com",
        "your email is ~aditya@rediff.com", "here is email ~bce@mail.com",
        "here is email id ~hello@find.in"
    ]
    s = ''
    for a in l:
        s += "[({},{},'EMAIL')]\n".format(
            re.search(r'~', a).start() + 1, len(a))
    print(s)


# new_train_gen()


def new_sp_model():
    TRAIN_DATA = [(u"my no is 9003401119", {
        "entities": [(9, 19, "PHONE")]
    }), (u"is your phone 9341234441", {
        "entities": [(14, 24, "PHONE")]
    }), (u"your phone number is 8341934568", {
        "entities": [(17, 27, "PHONE")]
    }), (u"here is my no 8261348649", {
        "entities": [(14, 24, "PHONE")]
    }), (u"here is my number 6713401897", {
        "entities": [(14, 24, "PHONE")]
    }),
        (u"my email is msubhaditya@gmail.com", {
            "entities": [(12, 34, "EMAIL")]
        }),
        (u"is your email id rules@yahoo.com", {
            "entities": [(17, 33, "EMAIL")]
        }),
        (u"your email is aditya@rediff.com", {
            "entities": [(14, 32, "EMAIL")]
        }),
        (u"here is email bce@mail.com", {
            "entities": [(14, 27, "EMAIL")]
        }),
        (u"here my email id hello@find.in", {
            "entities": [(17, 31, "EMAIL")]
        })]
    nlp = spacy.blank('en')
    # optimizer = nlp.begin_training()
    # for i in range(20):
    #     random.shuffle(TRAIN_DATA)
    #     for text, annotations in TRAIN_DATA:
    #         nlp.update([text], [annotations], sgd=optimizer)

    batches = spacy.util.minibatch(TRAIN_DATA)
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(texts, annotations)
    nlp.to_disk("newmod")


# new_sp_model()


# final context returns for csv
def context_json(p):
    dic = json.loads(return_context(p))
    d_final = {'persons': [], 'phone': [], 'emails': [], 'date': []}
    d_final['phone'].extend(re.findall(r'\d{10}', p))
    d_final['emails'].extend(re.findall(r'\S+@\S+', p))

    for a in dic:
        if dic[a] == 'PERSON':
            d_final['persons'].append(a)
        if dic[a] == 'DATE':
            d_final['date'].append(a)
    l = []
    for a in d_final:
        l.append(d_final[a])

    pd.DataFrame({k: pd.Series(l) for k, l in d_final.items()}).to_csv('output.csv',columns = ['persons','phone','emails','date'])
    # print(json.dumps(d_final))
    return json.dumps(d_final)


def calendar_entry():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server()
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('calendar', 'v3', credentials=creds)

    with open('output.csv','r') as f:
        p = f.readlines()[1].split(',')
        ev = p[-1].strip()
        su = p[1].strip()
        # print(ev)
        evp = datetime.datetime.strptime(ev, '%d %B %Y')
        ev = str(evp).split(' ')[0]+'T09:00:00-07:00'
        ev2 =str(evp).split(' ')[0]+'T19:00:00-07:00'
        # print(ev)
    event = {
    'summary': 'Added from AECP: {}'.format(su),
    'start':{
        'dateTime':str(ev),
    },
    'end':{
        'dateTime':str(ev2),
    },

    }
    event = service.events().insert(calendarId='primary', body=event).execute()

# calendar_entry()
