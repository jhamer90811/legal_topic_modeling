"""
Author: Jesse Hamer
Date: 6/25/19

This is a Python script designed for topic modeling of legal texts.

Command line inputs (must be specified in this order):

<input_dir>: The directory containing the input corpus
<vizualize>: either 'y' or 'n'; whether or not to use pyLDAvis to interactively inspect the learned topics. WARNING: depending on size of input corpus and number of topics in the model, this may take some time to run
<validate>: either 'y' or 'n'; whether or not to compute perplexity and coherence on the input set. Alternatively, if a directory is supplied which contains txt files, this will be used as a test corpus for validation.
<output_dir>: path to directory to save topic model; Use "None" if you do not wish to save the model

The pipeline expects the input corpus to be stored as raw txt files in a user-supplied directory.

This pipeline uses spacy to perform tokenization, lemmatization, removal of stopwords, and part-of-speech tagging to retain only nouns and verbs. (The preprocessor using only nouns achieved slightly higher coherence, but on inspection of wordclouds it became clear that many meaningful words were being left out of topics; e.g. "divorce" is typically tagged as a verb. The nouns+verbs preprocessor also performed better on fewer-topic models than did the nouns-only preprocessor.) Additionally, a list of 39 legal stopwords are pruned from the texts. The preprocessed documents are then fed into Gensim's LDA model with 50 topics (using the more parsimonious 25-topic model may also be preferable, as coherence was not too much lower than the 50-topic model).

When trained on a dataset of roughly 34k opinions randomly sampled from the jurisdiction of Illinois and dated after 1950, this model achieves coherence of 0.53 on a test set of roughly 8k opinions subject to the same conditions. The model also achieved a comparatively low perplexity of 136 on the test set.
"""

import os
import sys

import spacy

from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from gensim.models import CoherenceModel

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim 

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


input_dir = sys.argv[1]
if len(sys.argv)>2:
    vizualize = sys.argv[2]
else:
    visualize = 'n'
    
if len(sys.argv)>3:
    validate = sys.argv[3]
else:
    visualize = 'n'
    
if len(sys.argv)>4:
    output_dir = sys.argv[4]
else:
    output_dir = 'None'
    

print('Loading files from {}...'.format(input_dir))

opinions = []

for file in os.listdir(input_dir):
    with open(os.path.join(input_dir, file)) as f:
        opinions.append(f.read())
print('#####################')
print('Loading spacy model...')
print('#####################')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

basic_legal_stopwords = {'a.',
 'a.2d',
 'a.3d',
 'appeal',
 'appellant',
 'appellee',
 'case',
 'cir',
 'court',
 'defendant',
 'f. supp.',
 'f.supp.',
 'f.supp.2d',
 'f.supp.3d',
 'fact',
 'find',
 'hold',
 'judgment',
 'n.e.',
 'n.e.2d',
 'opinion',
 'order',
 'p.',
 'p.2d',
 'p.3d',
 'plaintiff',
 'question',
 's.e.',
 's.e.2d',
 's.e.3d',
 's.w.',
 's.w.2d',
 's.w.3d.',
 'see',
 'so.',
 'so.2d',
 'state',
 'time',
 'trial'}

print('#####################')
print('Preprocessing text...')
print('#####################')

docs = []
for op in opinions:
    doc = nlp(op)
    doc = [t.lemma_ for t in doc if (t.pos_ in ['NOUN','VERB']) and\
                                    (not t.is_stop) and\
                                    (t.lemma_ not in basic_legal_stopwords)]
    docs.append(doc)

opinions = None

print('#####################')
print('Training LDA model...')
print('#####################')

op_dictionary = Dictionary(docs)
op_corpus = [op_dictionary.doc2bow(doc) for doc in docs]

lda = LdaModel(op_corpus, id2word=op_dictionary, num_topics=50)

if validate !='n':
    
    if validate=='y':
        print('#####################')
        print('Validating LDA model on input data...')
        print('#####################')
        
        print('Perplexity: ', 2**(-lda.log_perplexity(op_corpus)))

        coherence_model = CoherenceModel(model=lda, texts=docs, dictionary=op_dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        print('Coherence Score: ', coherence)
    else:
        print('#####################')
        print('Validating LDA model on data in specified directory...')
        print('#####################')
        test_opinions = []

        for file in os.listdir(validate):
            with open(os.path.join(validate, file)) as f:
                test_opinions.append(f.read())
        test_docs = []
        for op in test_opinions:
            doc = nlp(op)
            doc = [t.lemma_ for t in doc if (t.pos_ in ['NOUN','VERB']) and\
                                            (not t.is_stop) and\
                                            (t.lemma_ not in basic_legal_stopwords)]
            test_docs.append(doc)
        test_opinions = None
        test_op_corpus = [op_dictionary.doc2bow(doc) for doc in test_docs]
        print('Perplexity: ', 2**(-lda.log_perplexity(test_op_corpus)))

        coherence_model = CoherenceModel(model=lda, texts=test_docs, dictionary=op_dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        print('Coherence Score: ', coherence)
    
# for topic in  lda.show_topics(num_words=1000, formatted=False):
#    topic_dict = {w:v for (w,v) in topic[1]}
#
#    wordcloud = WordCloud(width = 800, height = 200, 
#                    background_color ='white',
#                    min_font_size = 10).generate_from_frequencies(topic_dict) 
#
#    # plot the WordCloud image                        
#    plt.figure(figsize = (8, 8), facecolor = None) 
#    plt.imshow(wordcloud) 
#    plt.axis("off") 
#    plt.tight_layout(pad = 0) 
#
#    plt.show() 

if visualize ='y':
    print('#####################')
    print('Visualizing Topics...')
    print('#####################')
    vis = pyLDAvis.gensim.prepare(lda, op_corpus, op_dictionary)
    print('#####################')
    print('pyLDAvis prepared; opening browser...')
    print('#####################')
    pyLDAvis.show(vis)
    
if output_dir != 'None':
    print('#####################')
    print('Saving dictionary...')
    print('#####################')
    op_dictionary.save(os.path.join(output_dir, 'dictionary'))
    
    print('#####################')
    print('Saving topic model...')
    print('#####################')
    lda.save(os.path.join(output_dir, 'lda_model')
    