# Information-Retrieval

## Abstract:
In this project, a parser and an inverter was made to parse HTML pages and create inverted index. Four search algorithms (Okapi-TF, Okapi-TFIDF, Okapi-BM25 and Language Model with Jelinek Mercer Smoothing) were also implemented for document retrieval.

## How to run:
Files should be run in the following order

### parser.py
* python parser.py \<folder containing HTML files\>
* uses stoplist.txt, files in folder (contains HTML files) provided while execution
* creates docids.txt, termids.txt, doc_index.txt

### inverter.py
* python inverter.py
* uses docids.txt, termids.txt, doc_index.txt
* creates term_info.txt, term_index.txt

### docLengthCalculator.py
* python docLengthCalculator.py
* uses doc_index.txt
* creates doc_lengths.txt

### query.py
* python query.py --score \<score function\> --query \<search query\>
* available score functions: TF, TF-IDF, BM25, JM
* uses docids.txt, termids.txt, stoplist.txt, term_index.txt, doc_lengths.txt


## Contact
You can get in touch with me on my LinkedIn Profile: [Farhan Shoukat](https://www.linkedin.com/in/farhan-shoukat-782542167/)


## License
[MIT](../master/LICENSE)
Copyright (c) 2018 Farhan Shoukat
