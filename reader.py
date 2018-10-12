from argparse import ArgumentParser
from nltk.stem import PorterStemmer
import pandas as pd

OFFSET = 'Offset'
FREQUENCY = 'Frequency'
DOC_OCCURRENCES = 'Document Occurrences'

doc_ids = pd.read_csv('docids.txt', sep='\t', dtype=str, header=None, index_col=1).to_dict()[0]
term_ids = pd.read_csv('termids.txt', sep='\t', dtype=str, header=None, index_col=1).to_dict()[0]
term_info = pd.read_csv('term_info.txt', sep='\t', dtype=str, header=None, names=(OFFSET, FREQUENCY, DOC_OCCURRENCES),
                        index_col=0)
terms_index = open('term_index.txt', encoding='cp1252')


def print_doc_info(doc_name):
    try:
        doc_id = doc_ids[doc_name]
    except KeyError:
        print(doc_name, 'not found.')
        return None

    with open('doc_index.txt', encoding='cp1252') as f:
        doc_indexes = f.read().split('\n')

    index = 0
    for i, doc_index in enumerate(doc_indexes):
        if doc_index.split('\t')[0] == doc_id:
            index = i
            break

    distinct_terms = 0
    total_terms = 0
    for i in range(index, len(doc_indexes)):
        doc_index = doc_indexes[i].split('\t')
        if doc_index[0] == doc_id:
            distinct_terms += 1
            total_terms += len(doc_index) - 2
        else:
            break

    print('Listing for document:', doc)
    print('DOCID:', doc_id)
    print('Distinct terms:', distinct_terms)
    print('Total terms:', total_terms)


def print_term_info(term_name):
    term_name = PorterStemmer().stem(term_name)
    try:
        term_id = term_ids[term_name]
    except KeyError:
        print(term_name, 'not found.')
        return None

    print('Listing for term:', term_name)
    print('TERMID:', term_id)
    print('Number of documents containing term:', term_info.loc[term_id, DOC_OCCURRENCES])
    print('Term frequency in corpus:', term_info.loc[term_id, FREQUENCY])
    print('Inverted list offset:', term_info.loc[term_id, OFFSET])


def print_doc_term_info(doc_name, term_name):
    try:
        doc_id = doc_ids[doc_name]
    except KeyError:
        print(doc, 'does not exit.')
        return None

    term_name = PorterStemmer().stem(term_name)
    try:
        term_id = term_ids[term_name]
    except KeyError:
        print(term, 'does not exit.')
        return None

    terms_index.seek(int(term_info.loc[term_id, OFFSET]))
    term_index = terms_index.readline().split('\t')[1:]

    term_index = [[int(x.split(':')[0]), int(x.split(':')[1])] for x in term_index]
    doc_id = int(doc_id)
    positions = []
    if term_index[0][0] == doc_id:
        positions.append(str(term_index[0][1]))
    for i in range(1, len(term_index)):
        n = term_index[i][0] + term_index[i - 1][0]
        if n > doc_id:
            break
        if term_index[i][0] == 0:
            term_index[i][1] += term_index[i - 1][1]
        term_index[i][0] = n
        if n == doc_id:
            positions.append(str(term_index[i][1]))

    print('Inverted list for term:', term_name)
    print('In document:', doc_name)
    print('TERMID:', term_id)
    print('DOCID:', doc_id)
    print('Term frequency in document:', len(positions))
    print('Positions:', ', '.join(positions))


parser = ArgumentParser()
parser.add_argument('--doc', dest='doc', help='document title', metavar='DOC')
parser.add_argument('--term', dest='term', help='term/word', metavar='TERM')
options = parser.parse_args()
doc = options.doc
term = options.term

if doc is not None and term is not None:
    print_doc_term_info(doc, term)
elif doc is not None:
    print_doc_info(doc)
elif term is not None:
    print_term_info(term)
else:
    print('No arguments provided.')
    print('Provide either document or term or both with --doc <document title> or --term <term> respectively.')

terms_index.close()
