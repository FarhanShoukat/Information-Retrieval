import sys
import re
from os import listdir
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


directory = None
documents = None
stop_words = None

try:
    directory = sys.argv[1]
    documents = listdir(directory)
except IndexError:
    print('Directory not provided. Provide it like "python tokenizer.py <Directory>". Program exiting.')
    exit()
except FileNotFoundError:
    print('Directory not Found. program exiting.')
    exit()
directory = directory + '/'

try:
    with open('stoplist.txt') as f:
        print('Reading Stop List.')
        stop_words = f.read().split('\n')
        print('Stop List read.')
except FileNotFoundError:
    print('stoplist.txt not found. program exiting.')
    exit()


# reading html data ###################################################################################################
print('\nReading Data.')
data = []
not_read = []
not_html = []
for document in documents:
    try:
        with open(directory + document) as fileHandle:
            html = fileHandle.read().lower()
    except UnicodeDecodeError:
        try:
            with open(directory + document, encoding='iso8859-1') as fileHandle:
                html = fileHandle.read().lower()
        except UnicodeDecodeError:
            not_read.append(document)
            continue
    position = html.find('\n<')
    if position == -1:
        not_html.append(document)
        html = '\n'.join(html.split('\n')[20:])
    else:
        html = html[position + 1:]
    data.append(html)

print('Data Read.')
if len(not_read) != 0:
    print('Files failed to read:', not_read, '\nRemoving them from Corpus.')
    documents = [document for document in documents if document not in not_read]
if len(not_html) != 0:
    print('Files not html:', not_html)
not_html = None


# parsing and pre-processing html #####################################################################################
print('\nParsing and Pre-processing Data.')
not_read.clear()
terms = set()
stemmer = PorterStemmer()
regex = re.compile('[^a-z0-9 ]')
stop_words = set(stop_words + [stemmer.stem(stop_word) for stop_word in stop_words])
stop_words.add('')
for i, html in enumerate(data):
    # get text from html
    try:
        parsed_html = BeautifulSoup(html, features='html5lib')
    except UserWarning:
        not_read.append(i)
        continue
    for script in parsed_html(['script', 'style']):
        script.decompose()  # rip it out
    try:
        text = ' '.join(parsed_html.strings).replace('\n', ' ')
    except AttributeError:
        not_read.append(i)
        continue

    # tokenize, stem, remove stop-words
    tokens = re.split('\W+', text)
    filtered_tokens = [token for token in
                       [stemmer.stem(token) for token in tokens]
                       if token not in stop_words]

    # adding newly found words to unique terms set
    terms |= set(filtered_tokens)

    data[i] = filtered_tokens

print('Data parsed and Preprocessed.')
if len(not_read) != 0:
    print('Files failed to Parse & Pre-process:', [documents[x] for x in not_read], '\nRemoving them from Corpus.')
    for i in not_read:
        del documents[i]
        del data[i]
not_read = None

# converting terms set to list
terms = sorted(terms)


# create termids.txt file #############################################################################################
print('\nCreating Files.')
to_write = '\n'.join([str(i + 1) + '\t' + term for i, term in enumerate(terms)])
with open('termids.txt', 'w', encoding='utf8') as termID:
    termID.write(to_write)
    print('termids.txt created.')

# create docids.txt file
to_write = '\n'.join([str(i + 1) + '\t' + document for i, document in enumerate(documents)])
with open('docids.txt', 'w', encoding='cp1252') as docID:
    docID.write(to_write)
    print('docids.txt created.')

# create doc_index.txt
terms = {v: k + 1 for k, v in enumerate(terms)}
to_write = []
for i, d in enumerate(data):
    term_positions = {}
    doc_terms = []
    for j, token in enumerate(d):
        try:
            term_positions[token].append(str(j + 1))
        except KeyError:
            term_positions[token] = [str(j + 1)]
            doc_terms.append(token)
    for u in doc_terms:
        to_write.append(str(i + 1) + '\t' + str(terms[u]) + '\t' + '\t'.join(term_positions[u]))

to_write = '\n'.join(to_write)
with open('doc_index.txt', 'w', encoding='cp1252') as doc_index:
    doc_index.write(to_write)
    print('doc_index.txt created.')
