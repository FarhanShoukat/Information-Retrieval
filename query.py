from argparse import ArgumentParser
from nltk.stem import PorterStemmer
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from itertools import islice
import statistics
import time

OFFSET = 'Offset'
FREQUENCY = 'Frequency'
DOC_OCCURRENCES = 'Document Occurrences'
k1 = 1.2
k2 = 500
b = 0.9


def get_term_posting(term):
    terms_index.seek(int(term_info.loc[term, OFFSET]))
    term_posting = terms_index.readline().rstrip().split('\t')

    # parsing posting list of term into numpy array of n * 2 where n is the total occurrence of term in corpus
    return np.array([[int(doc), int(pos)] for doc, pos in
                     [x.split(':') for x in islice(term_posting, 1, len(term_posting))]])


def query_preprocessing(query):
    # lowered, tokenized, stemmed, stop words removed
    query = [term for term in
             [stemmer.stem(term) for term in regex.sub('', query.lower()).split()]
             if term not in stop_words]

    # each term is converted to its id
    q = []
    for term in query:
        try:
            q.append(term_ids[term])
        except KeyError:
            pass
    return ' '.join(q)


# create count vectors of query and documents related to query
def create_count_vectors(query):
    vectorizer = CountVectorizer()

    # creating count vector of query
    query_vector = vectorizer.fit_transform([query]).toarray()[0]
    features = vectorizer.get_feature_names()

    documents = {}
    for term in features:
        term_posting = get_term_posting(term)

        # adding first position of posting list to respective document
        try:
            documents[str(term_posting[0, 0])].append(term)
        except KeyError:
            documents[str(term_posting[0, 0])] = [term]

        # delta decoding and adding terms to respective document list
        for i in range(1, len(term_posting)):
            # if term_posting[i, 0] == 0:
            #     term_posting[i, 1] += term_posting[i - 1, 1]
            term_posting[i, 0] += term_posting[i - 1, 0]
            try:
                documents[str(term_posting[i, 0])].append(term)
            except KeyError:
                documents[str(term_posting[i, 0])] = [term]

    # joining lists of documents containing term ids and converting dictionary to list for vectorizer
    doc_references = {}
    d = []
    for i, key in enumerate(documents):
        d.append(' '.join(documents[key]))
        doc_references[key] = i
    documents = d

    # creating count vector of documents
    documents_vectors = vectorizer.transform(documents).toarray()

    return query_vector, doc_references, documents_vectors, features


def get_okapi_tf_vector(vector, doc_len): return vector / (vector + 0.5 + 1.5 * doc_len / AVG_DOC_LEN)


def get_okapi_tf_idf_vector(vector, doc_len, log_d_by_df): return get_okapi_tf_vector(vector, doc_len) * log_d_by_df


def okapi_tf(query):
    query_vector, doc_references, doc_vectors, features = create_count_vectors(query)
    reference_to_doc = {reference: int(doc) for doc, reference in doc_references.items()}

    # creating okapi-tf vectors of query and documents
    query_vector = get_okapi_tf_vector(query_vector, len(query.split()))
    doc_vectors = [get_okapi_tf_vector(doc_vectors[i], doc_lengths[reference_to_doc[i]])
                   for i in range(len(doc_vectors))]

    # finding cosine similarity scores of query with documents
    query_vector_len = np.sqrt(query_vector.dot(query_vector))
    doc_scores = [query_vector.dot(doc_vector) / (query_vector_len * np.sqrt(doc_vector.dot(doc_vector)))
                  for doc_vector in doc_vectors]

    return doc_references, doc_scores


def okapi_tf_idf(query):
    query_vector, doc_references, doc_vectors, features = create_count_vectors(query)
    reference_to_doc = {reference: int(doc) for doc, reference in doc_references.items()}

    df = np.array([int(term_info.loc[feature, DOC_OCCURRENCES]) for feature in features])
    log_d_by_df = np.log10(DOC_COUNT / df)

    # creating okapi-tf vectors of query and documents
    query_vector = get_okapi_tf_idf_vector(query_vector, len(query.split()), log_d_by_df)
    doc_vectors = [get_okapi_tf_idf_vector(doc_vectors[i], doc_lengths[reference_to_doc[i]], log_d_by_df)
                   for i in range(len(doc_vectors))]
    del reference_to_doc

    # finding cosine similarity scores of query with documents
    query_vector_len = np.sqrt(query_vector.dot(query_vector))
    doc_scores = [query_vector.dot(doc_vector) / (query_vector_len * np.sqrt(doc_vector.dot(doc_vector)))
                  for doc_vector in doc_vectors]

    return doc_references, doc_scores


# parser = ArgumentParser()
# parser.add_argument('--score', dest='score', help='name of scoring function (TF or TF-IDF)',
#                     metavar='SCORE', required=True)
# options = parser.parse_args()
# score_function = options.score.lower()
# if score_function == 'tf':
#     score_function = okapi_tf
# elif score_function == 'tf-idf':
#     score_function = okapi_tf_idf
# else:
#     print('Please select valid score function')
#     exit(-1)

doc_ids = pd.read_csv('docids.txt', sep='\t', dtype=str, header=None, index_col=0).to_dict()[1]
doc_lengths = pd.read_csv('doc_lengths.txt', sep='\t', dtype=int, header=None, index_col=0).to_dict()[1]
term_ids = pd.read_csv('termids.txt', sep='\t', dtype=str, header=None, index_col=1).to_dict()[0]
term_info = pd.read_csv('term_info.txt', sep='\t', dtype=str, header=None, names=(OFFSET, FREQUENCY, DOC_OCCURRENCES),
                        index_col=0)
terms_index = open('term_index.txt', encoding='cp1252')

with open('topics.xml') as f:
    topics = f.read()

with open('stoplist.txt') as f:
    stop_words = f.read().split('\n')

AVG_DOC_LEN = statistics.mean(doc_lengths.values())
DOC_COUNT = len(doc_ids)

stemmer = PorterStemmer()
stop_words = set(stop_words + [stemmer.stem(stop_word) for stop_word in stop_words])
regex = re.compile('[^a-z0-9 ]')
topics = BeautifulSoup(topics, features='html5lib').find_all('topic')
topics = [(topic['number'], query_preprocessing(topic.find('query').getText())) for topic in topics]

# for number, topic in topics:
#     score_function(topic)

terms_index.close()
