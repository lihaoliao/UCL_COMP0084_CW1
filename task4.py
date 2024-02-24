from collections import defaultdict
import json
import csv
import time


start_time = time.time()

qid_and_pid = defaultdict(set)
with open('qid_and_pid.txt', 'r') as file:
    for line in file:
        qid, pids_str = line.strip().split(': ')
        qid_and_pid[qid].update(pids_str.split(','))       

# 这个是tf的分子,就是每个term在某个passage出现的次数 value长度就是每个单词出现的总次数
with open('inverted_index.json', 'r', encoding='utf-8') as file:
    inverted_index = json.load(file)

# 这个变量长度（总共的document数量）是idf分子 然后它每个passage的value是number of occurrences of words = D
with open('each_passage_terms_sum.json', 'r', encoding='utf-8') as file:
    each_passage_terms_sum = json.load(file)

# len = number of unique words in the entire collection (vocabulary size)
with open('remove_stop_word_vocabulary.txt', 'r', encoding='utf-8') as file:
        vocabulary_from_task1 = [line.strip() for line in file.readlines()]
vocabulary_size = len(vocabulary_from_task1)        

with open('queries_id_and_terms_info.json', 'r', encoding='utf-8') as file:
    queries_id_and_terms_info = json.load(file) 

with open('passages_id_and_terms_info.json', 'r', encoding='utf-8') as file:
    passages_id_and_terms_info = json.load(file)

qid_and_pid = defaultdict(set)
with open('candidate-passages-top1000.tsv', 'r', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
        qid,pid = row[0],row[1]
        if qid not in qid_and_pid[qid]:
            qid_and_pid[qid].add(pid)                        

def calcaulate_Laplace(queries_id_and_terms_info, inverted_index, vocabulary_size, passages_id_and_terms_info, qid_and_pid):
    #  passage : score
    laplace_estimates_scores =  {}
    for qid, pids in qid_and_pid:
        query_words = queries_id_and_terms_info[qid]
        for pid in pids:
            for term in query_words:
            # query - passage - score
            probabilities_of_word = 1


laplace_estimates_scores = calcaulate_Laplace(queries_id_and_terms_info, inverted_index, passages_id_and_terms_info,qid_and_pid)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"task3程序运行时间：{elapsed_time}秒")    