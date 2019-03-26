import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
from nltk import ne_chunk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import re
import heapq
import json

f = open('convotext.txt', 'r',errors='ignore').read().lower()

    #     f = re.sub(r'\s+', ' ', f)
no_of_lines = len(open('convotext.txt', 'r',errors='ignore').readlines())
stop_words = set(stopwords.words('english'))

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
    filtered_sentence = [w for w in sent.split(' ') if not w in stop_words]

    return ' '.join(filtered_sentence)


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

def extractive_summary(f,docu):
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
    doc = nlp(docu)
    fin_dic = {}
    for ent in doc.ents:
        fin_dic[ent.text]=ent.label_
    return json.dumps(fin_dic,sort_keys=True)



nlp = en_core_web_sm.load()
docu = preprocess(f)

# return_context(docu)
extractive_summary(docu)
