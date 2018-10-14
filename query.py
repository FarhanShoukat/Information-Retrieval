from argparse import ArgumentParser
from nltk.stem import PorterStemmer
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from itertools import islice

OFFSET = 'Offset'
FREQUENCY = 'Frequency'
DOC_OCCURRENCES = 'Document Occurrences'


# create count vectors of query and documents related to query
def create_vectors(query):
    vectorizer = CountVectorizer()
    # lowered, tokenized, stemmed
    query = [stemmer.stem(term) for term in regex.sub('', query.lower()).split()]

    # stop words removal
    query = [term for term in query if term not in stop_words]

    # each term is converted to its id
    q = []
    for term in query:
        try:
            q.append(term_ids[term])
        except KeyError:
            pass
    query = ' '.join(q)

    # creating count vector of query
    query_vector = vectorizer.fit_transform([query]).toarray()

    unique_terms = vectorizer.get_feature_names()

    terms_posting = []
    for term in unique_terms:
        terms_index.seek(int(term_info.loc[term, OFFSET]))
        term_posting = terms_index.readline().rstrip().split('\t')
        term_posting = np.array([[int(doc), int(pos)] for doc, pos in [x.split(':') for x in
                                                                       islice(term_posting, 1, len(term_posting))]])
        for i in range(1, len(term_posting)):
            if term_posting[i, 0] == 0:
                term_posting[i, 1] += term_posting[i - 1, 1]
            term_posting[i, 0] += term_posting[i - 1, 0]


def okapi_tf(topic):
    print('okapi:', topic)


def tf_tdf(topic):
    print('tf-idf:', topic)


# parser = ArgumentParser()
# parser.add_argument('--score', dest='score', help='name of scoring function (TF or TF-IDF)', metavar='SCORE', required=True)
# options = parser.parse_args()
# score_function = options.score.lower()
# if score_function != 'tf-idf' and score_function != 'tf':
#     print('Please select valid score function')

doc_ids = pd.read_csv('docids.txt', sep='\t', dtype=str, header=None, index_col=1).to_dict()[0]
term_ids = pd.read_csv('termids.txt', sep='\t', dtype=str, header=None, index_col=1).to_dict()[0]
term_info = pd.read_csv('term_info.txt', sep='\t', dtype=str, header=None, names=(OFFSET, FREQUENCY, DOC_OCCURRENCES),
                        index_col=0)
terms_index = open('term_index.txt', encoding='cp1252')

with open('topics.xml') as f:
    topics = f.read()

with open('stoplist.txt') as f:
    stop_words = f.read().split('\n')

stemmer = PorterStemmer()
stop_words = set(stop_words + [stemmer.stem(stop_word) for stop_word in stop_words])
regex = re.compile('[^a-z0-9 ]')
topics = BeautifulSoup(topics, features='html5lib')
topics = [topic.getText() for topic in topics.find_all('query')]

create_vectors(topics[0])
# if score_function == 'tf-idf':
#     tf_tdf(topics)
# elif score_function == 'tf':
#     okapi_tf(topics)

terms_index.close()