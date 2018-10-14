from argparse import ArgumentParser
from nltk.stem import PorterStemmer


def get_id(file_name, name):
    try:
        with open(file_name, encoding='cp1252') as docs_info:
            docs = docs_info.read().split('\n')
    except FileNotFoundError:
        # print(file_name, 'not found. Program exiting.')
        return -1
    for d in docs:
        d = d.split('\t')
        if d[1] == name:
            return d[0]


def get_doc_info(doc_id):
    try:
        with open('doc_index.txt', encoding='cp1252') as f:
            doc_indexes = f.read().split('\n')
    except FileNotFoundError:
        return -1

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
            return distinct_terms, total_terms


def get_term_info(term_id):
    try:
        with open('term_info.txt', encoding='cp1252') as f:
            return f.read().split('\n')[int(term_id) - 1].split('\t')
    except FileNotFoundError:
        # print('term_info.txt not found. Program exiting.')
        return -1


def doc_info(doc_name):
    doc_id = get_id('docids.txt', doc_name)
    if doc_id == -1:
        print('docids.txt not found.')
        return None
    elif doc_id is None:
        print(doc_name, 'not found.')
        return None
    else:
        distinct_terms, total_terms = get_doc_info(doc_id)
        print('Listing for document:', doc)
        print('DOCID:', doc_id)
        print('Distinct terms:', distinct_terms)
        print('Total terms:', total_terms)


def term_info(term_name):
    term_name = PorterStemmer().stem(term_name)
    term_id = get_id('termids.txt', term_name)
    if term_id == -1:
        print('termids.txt not found.')
        return None
    elif term_id is None:
        print(term_name, 'not found.')
        return None
    else:
        info = get_term_info(term_id)
        print('Listing for term:', term_name)
        print('TERMID:', term_id)
        print('Number of documents containing term:', info[3])
        print('Term frequency in corpus:', info[2])
        print('Inverted list offset:', info[1])


def doc_term_info(doc_name, term_name):
    doc_id = get_id('docids.txt', doc_name)
    if doc_id == -1:
        print('docids.txt not found.')
        return None
    elif doc_id is None:
        print(doc, 'does not exit.')
        return None

    term_name = PorterStemmer().stem(term_name)
    term_id = get_id('termids.txt', term_name)
    if term_id == -1:
        print('termids.txt not found.')
        return None
    elif term_id is None:
        print(term, 'does not exit.')
        return None

    term_offset = get_term_info(term_id)[1]
    try:
        with open('term_index.txt', encoding='cp1252') as f:
            f.seek(int(term_offset))
            term_index = f.readline().split('\t')[1:]
    except FileNotFoundError:
        print('term_index.txt not found.')
        return None

    term_index = [[int(x.split(':')[0]), int(x.split(':')[1])] for x in term_index]
    doc_id = int(doc_id)
    positions = []
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
    doc_term_info(doc, term)
elif doc is not None:
    doc_info(doc)
elif term is not None:
    term_info(term)
else:
    print('No arguments provided.')
    print('provide either document or term or both with --doc <document title> or --term <term> respectively')
