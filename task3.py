from collections import Counter, defaultdict
import os
import json
import math
import re
import time
import csv

import numpy as np

start_time = time.time()
# 这个是tf的分子,就是每个term在某个passage出现的次数
with open('inverted_index.json', 'r', encoding='utf-8') as file:
    inverted_index = json.load(file)

# 这个变量长度是idf分子 然后它每个passage的value是tf的分母(也就是passage一共有几个词)
with open('each_passage_terms_sum.json', 'r', encoding='utf-8') as file:
    each_passage_terms_sum = json.load(file)

# 这个是passage和对应的terms
with open('passages_id_and_terms_info.json', 'r', encoding='utf-8') as file:
    passages_id_and_terms_info = json.load(file)    

total_passage = len(each_passage_terms_sum)

# calculate IDF for each term - shared by query and passage
idf = {term: math.log10(total_passage/len(inverted_index[term])) for term in inverted_index}

# calculate the tf for each term in each passage(query)
def calculate_passage_tf(inverted_index,each_passage_terms_sum):
    term_tf_in_passage = defaultdict(dict)
    for term, passages in inverted_index.items():
        for pid, term_frequency in passages.items():
            total_terms_in_passage = each_passage_terms_sum[pid]
            tf = term_frequency / total_terms_in_passage
            if term not in term_tf_in_passage[term]:
                term_tf_in_passage[term][pid] = tf
    return term_tf_in_passage                  

term_tf_in_passage = calculate_passage_tf(inverted_index,each_passage_terms_sum)

# calculate the tf-idf for each term in each passage(query)
def calculate_passage_tf_idf(term_tf_in_passage, idf, passages_id_and_terms_info):
    tf_idf_for_passage = defaultdict(dict)
    for pid, terms in passages_id_and_terms_info.items():
        tf_idf_cur_passage = {}
        for term in terms:
            if term not in tf_idf_cur_passage:
                tf_idf_cur_passage[term] = term_tf_in_passage[term][pid] * idf[term]

        tf_idf_for_passage[pid] =  tf_idf_cur_passage      
    return tf_idf_for_passage    
tf_idf_for_passage =  calculate_passage_tf_idf(term_tf_in_passage, idf, passages_id_and_terms_info)

preprocessing_re = re.compile(r'[^a-zA-Z\s]')
def read_remove_stop_word_vocabulary_from_task1(filename='remove_stop_word_vocabulary.txt'):
  
    with open(filename, 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file.readlines()]
    
    return words
def preprocessing_query_passage(text):
    text = text.replace("/", " ")
    text = re.sub(preprocessing_re, '', text)
    return text

queries_id_and_terms_info = {}
vocabulary_from_task1 = set(read_remove_stop_word_vocabulary_from_task1())

with open('test-queries.tsv', 'r', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
        row[1] = preprocessing_query_passage(row[1])
        qid, query = row[0], row[1]
        tokens = query.split()
        query_contain_token = [token for token in tokens if token in vocabulary_from_task1]
        if query_contain_token:
            queries_id_and_terms_info[qid] = query_contain_token
        else:
            queries_id_and_terms_info[qid] = []    

with open('queries_id_and_terms_info.json', 'w', encoding='utf-8') as file:
    json.dump(queries_id_and_terms_info, file, ensure_ascii=False, indent=3)     

inverted_index_query = defaultdict(dict)   
each_query_terms_sum = {}     
total_query = len(queries_id_and_terms_info)
for qid, terms in queries_id_and_terms_info.items():
    term_frequency = Counter(terms)
    each_query_terms_sum[qid] = sum(term_frequency.values())
    for term, frequency in term_frequency.items():
        if qid not in inverted_index_query[term]:
            inverted_index_query[term][qid] = frequency

with open('inverted_index_query.json', 'w', encoding='utf-8') as file:
    json.dump(inverted_index_query, file, ensure_ascii=False, indent=3)    

# calculate the tf for each term in each query
term_tf_in_query = calculate_passage_tf(inverted_index_query,each_query_terms_sum)

# calculate the tf-idf for each term in each query
tf_idf_for_query =  calculate_passage_tf_idf(term_tf_in_query, idf, queries_id_and_terms_info)

# read the tsv to get the connection between qid and pid
qid_and_pid = defaultdict(set)
with open('candidate-passages-top1000.tsv', 'r', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
        qid,pid = row[0],row[1]
        if qid not in qid_and_pid[qid]:
            qid_and_pid[qid].add(pid)

# calculate the cosine_similarity
def calculate_cosine_similarity(tf_idf_for_passage,tf_idf_for_query):
    cosine_similarity_score = defaultdict(dict)
    for qid, pids in qid_and_pid.items():
        for pid in pids:
            qid_vector = []
            pid_vector = []
            term_tf_in_pid_union_qid = set(tf_idf_for_query[qid].keys()) | set(tf_idf_for_passage[pid].keys())
            for term in term_tf_in_pid_union_qid:
                qid_vector.append(tf_idf_for_query[qid].get(term, 0))
                pid_vector.append(tf_idf_for_passage[pid].get(term, 0))
            qid_vector = np.array(qid_vector)
            pid_vector = np.array(pid_vector)    
            inner_product = np.dot(pid_vector,qid_vector)
            norm_query = np.linalg.norm(qid_vector)
            norm_passage = np.linalg.norm(pid_vector)
            if norm_query != 0 and norm_passage != 0:
                cosine_similarity = inner_product / (norm_query * norm_passage)    
            else:
                cosine_similarity = 0    
            if pid not in cosine_similarity_score[qid]:
                cosine_similarity_score[qid][pid] = cosine_similarity        
    return cosine_similarity_score


cosine_similarity_score = calculate_cosine_similarity(tf_idf_for_passage,tf_idf_for_query)
with open('cosine_similarity.json', 'w', encoding='utf-8') as file:
    json.dump(cosine_similarity_score, file, ensure_ascii=False, indent=3)

# select the top 100
def tfidf_top100(cosine_similarity_score,queries_id_and_terms_info):
    top_100_pids_for_qid = defaultdict(list)
    for qid, pid_scores in cosine_similarity_score.items():
        sorted_pids = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
        top_100_pids = sorted_pids[:100]
        top_100_pids_for_qid[qid] = top_100_pids 
    with open('tfidf.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for qid, _ in queries_id_and_terms_info.items():
            top100_passage = top_100_pids_for_qid[qid]
            for pid,score in top100_passage:
                writer.writerow([qid, pid, score])

tfidf_top100(cosine_similarity_score,queries_id_and_terms_info)

# =========================================================================================================================

end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time}秒")
