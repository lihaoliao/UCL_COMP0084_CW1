import re
import time
import csv
import json
from collections import Counter,defaultdict

preprocessing_re = re.compile(r'[^a-zA-Z\s]')
def read_remove_stop_word_vocabulary_from_task1(filename='remove_stop_word_vocabulary.txt'):
  
    with open(filename, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file.readlines()]
    
    return words

def preprocessing_passage(text):
    text = text.replace("/", " ")
    text = re.sub(preprocessing_re, '', text)
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
            else:
                passages_id_and_terms_info[pid] = []     
              
inverted_index = defaultdict(dict)
total_passage = len(passages_id_and_terms_info)
for pid, terms in passages_id_and_terms_info.items():
    term_frequency = Counter(terms)
    for term, frequency in term_frequency.items():
        if pid not in inverted_index[term]:
            inverted_index[term][pid] = frequency

with open('inverted_index.json', 'w', encoding='utf-8') as file:
    json.dump(inverted_index, file, ensure_ascii=False, indent=3)

with open('passages_id_and_terms_info.json', 'w', encoding='utf-8') as file:
    json.dump(passages_id_and_terms_info, file, ensure_ascii=False, indent=3)  
       
end_time = time.time()
elapsed_time = end_time - start_time
print(f"task2程序运行时间：{elapsed_time}秒")