data = None
try:
    with open('doc_index.txt', encoding='cp1252') as doc_index:
        data = doc_index.read().split('\n')
        print()
except FileNotFoundError:
    print('doc_index.txt not found. Program exiting.')
    exit()

inverted_index = {}
corpus_occurrences = {}
document_occurrences = {}

for line in data:
    line = line.split('\t')
    doc_id = int(line[0])
    term_id = line[1]

    document_occurrences[term_id] = document_occurrences.get(term_id, 0) + 1
    corpus_occurrences[term_id] = corpus_occurrences.get(term_id, 0) + len(line) - 2

    # delta encoding positions of term in a document
    positions = [int(line[i]) for i in range(2, len(line))]
    for i in range(len(positions) - 1, 0, -1):
        positions[i] -= positions[i - 1]

    # adding document information i.e. document id and positions of that term to partial inverted index
    try:
        inverted_index[term_id].append([doc_id, positions])
    except KeyError:
        inverted_index[term_id] = [[doc_id, positions]]

term_ids = list(inverted_index)
term_ids.sort(key=int)

to_write = []
for term_id in term_ids:
    term_info = inverted_index[term_id]
    # delta encoding document ids
    for i in range(len(term_info) - 1, 0, -1):
        term_info[i][0] -= term_info[i - 1][0]

    s = [term_id]
    for doc_term_info in term_info:
        doc_id = str(doc_term_info[0])

        doc_term_info = doc_term_info[1]
        s.append(str(doc_id) + ':' + str(doc_term_info[0]))
        for i in range(1, len(doc_term_info)):
            s.append('0:' + str(doc_term_info[i]))
    to_write.append('\t'.join(s))


# creating byte offset of each term ###################################################################################
offset = 0
offsets = {}
for i, x in enumerate(to_write):
    offsets[term_ids[i]] = offset
    offset += len(x) + 2


# creating term_index.txt #############################################################################################
with open('term_index.txt', 'w', encoding='cp1252') as f:
    f.write('\n'.join(to_write))
print('term_index.txt is created.')

# joining termID, term offset in ter_index.txt, term count in corpus, document count of term
to_write = [str(i) + '\t' + str(offsets[i]) + '\t' + str(corpus_occurrences[i]) + '\t' + str(document_occurrences[i])
            for i in term_ids]

# creating term_info.txt
with open('term_info.txt', 'w', encoding='cp1252') as f:
    f.write('\n'.join(to_write))
print('term_info.txt is created.')
