with open('doc_index.txt') as f:
    content = f.readlines()

doc_len = {}
for line in content:
    line = line.split('\t')
    doc_len[line[0]] = doc_len.get(line[0], 0) + len(line) -2

to_write = '\n'.join([doc + '\t' + str(length) for doc, length in doc_len.items()])
with open('doc_length.txt', mode='w') as f:
    f.write(to_write)
