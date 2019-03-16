
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm


f =open('convotext.txt','r').read().lower()

def preprocess(sent):
    with tf.device('/device:GPU:0'):
        sent = nltk.word_tokenize(sent,language='english')
        sent = nltk.pos_tag(sent)
        return sent

tag = preprocess(f)
pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(tag)
print(cs)

iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)
ne_tree = ne_chunk(pos_tag(word_tokenize(ex)))

nlp = en_core_web_sm.load()
doc = nlp(f)
pprint([(X.text, X.label_) for X in doc.ents])