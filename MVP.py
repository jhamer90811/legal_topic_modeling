"""
Author: Jesse Hamer
Date: 6/13/19

This is a baseline topic model extraction script intended for demo purposes only.

Run from command line. Input should be a directory containing input documents.
"""

import os
import sys

import spacy

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from gensim.models import CoherenceModel

from wordcloud import WordCloud
import matplotlib.pyplot as plt
# import pyLDAvis
# import pyLDAvis.gensim 

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


input_dir = sys.argv[1]
print('Loading files from {}...'.format(input_dir))

opinions = []

for file in os.listdir(input_dir):
    with open(os.path.join(input_dir, file)) as f:
        opinions.append(f.read())
print('#####################')
print('Loading spacy model...')
print('#####################')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

print('#####################')
print('Preprocessing text...')
print('#####################')

docs = []
for op in opinions:
    doc = nlp(op)
    doc = [t for t in doc if not t.is_stop]
    doc = [t.lemma_ for t in doc if t.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]
    docs.append(doc)

print('#####################')
print('Training LDA model...')
print('#####################')

op_dictionary = Dictionary(docs)
op_corpus = [op_dictionary.doc2bow(doc) for doc in docs]

lda = LdaModel(op_corpus, id2word=op_dictionary, num_topics=5)

print('#####################')
print('Validating LDA model...')
print('#####################')

print('Perplexity: ', 2**(-lda.log_perplexity(op_corpus)))

coherence_model = CoherenceModel(model=lda, texts=docs, dictionary=op_dictionary, coherence='c_v')
coherence = coherence_model.get_coherence()
print('Coherence Score: ', coherence)
    
print('#####################')
print('Visualizing Topics...')
print('#####################')

for topic in  lda.show_topics(num_words=1000, formatted=False):
    topic_dict = {w:v for (w,v) in topic[1]}

    wordcloud = WordCloud(width = 800, height = 200, 
                    background_color ='white',
                    min_font_size = 10).generate_from_frequencies(topic_dict) 

    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show() 
    
# vis = pyLDAvis.gensim.prepare(lda, op_corpus, op_dictionary)
# pyLDAvis.show(vis)