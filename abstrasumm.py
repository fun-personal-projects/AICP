
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
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
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.show()
    plt.axis('off')
    plt.savefig('trends.png')

    return dict_trend.most_common(5)


# l = [
#     '''
# Good morning! Madam.

# Good morning. Sit down. What do you want?

# I want admission into Sixth Standard.

# Where did you study last year?

# I studied in Tirumangalam.

# Then why do you want admission here?

#  My father has been transferred to Madurai Branch.

# What is your father?

# He is a Bank Officer.

# Where is he?

# He is seated there. Shall I call him?

# Fill in this application form and come in the afternoon.
# ''', '''
# Good morning both of you. He is my friend Suresh.

# Good morning. Who is he?

# He is Rahul.

# Good Morning Suresh.

# Where are you studying Rahul?

# I am studying in St. Mary's High School.

# Do you come to school by cycle?

# No. I come to school on foot. What about you Suresh?

# I attend the school by bus.

# Do you like to witness cricket match?

# I am interested in watching one day matches.

# Very fine. We shall go to Racecourse grounds to watch one-day match.

# '''
# ]

# trends(l)

