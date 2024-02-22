import os
import re
import time
import csv
from collections import Counter,defaultdict

def read_remove_stop_word_vocabulary_from_task1(filename='remove_stop_word_vocabulary.txt'):
  
    with open(filename, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file.readlines()]

    # if os.path.exists(filename):
    #     os.remove(filename)
    
    return words

def preprocessing_passage(text):
    text = text.replace("/", " ")
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

start_time = time.time()
visited_ids = set()
passages_id_and_terms_info = {}
vocabulary_from_task1 = set(read_remove_stop_word_vocabulary_from_task1())

with open('candidate-passages-top1000.tsv', 'r', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
        row[3] = preprocessing_passage(row[3])
        pid, passage = row[1], row[3]
        if pid not in visited_ids:
            visited_ids.add(pid)
            # same method to tokenise as task 1
            tokens = passage.split()
            passage_contain_token = [token for token in tokens if token in vocabulary_from_task1]
            if passage_contain_token:
                passages_id_and_terms_info[pid] = passage_contain_token
              
inverted_index = defaultdict(dict)
passage_terms_sum = {}
total_passage = len(passages_id_and_terms_info)
for pid, terms in passages_id_and_terms_info.items():
    term_frequency = Counter(terms)
    passage_terms_sum[pid] = sum(term_frequency.values())
    for term, frequency in term_frequency.items():
        if pid not in inverted_index[term]:
            inverted_index[term][pid] = frequency
            
# doc_freqs = {term: len(doc_ids) for term, doc_ids in inverted_index.items()}                   
         
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time}秒")