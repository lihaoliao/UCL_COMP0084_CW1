from collections import Counter, defaultdict
import json
import re
import time
import csv

import numpy as np

start_time = time.time()

with open('inverted_index.json', 'r', encoding='utf-8') as file:
    inverted_index = json.load(file)
    
with open('passages_id_and_terms_info.json', 'r', encoding='utf-8') as file:
    passages_id_and_terms_info = json.load(file)    

total_passage = len(passages_id_and_terms_info)

# calculate IDF for each term - shared by query and passage
idf = {term: np.log10(total_passage/len(inverted_index[term])) for term in inverted_index}

# calculate the tf for each term in each passage/query
def calculate_tf(inverted_index,passages_or_queries_id_and_terms_info):
    term_tf_in_passage = defaultdict(dict)
    for term, passages in inverted_index.items():
        for pid, term_frequency in passages.items():
            total_terms = len(passages_or_queries_id_and_terms_info[pid])
            tf = term_frequency / total_terms
            if term not in term_tf_in_passage[term]:
                term_tf_in_passage[term][pid] = tf
    return term_tf_in_passage                  

term_tf_in_passage = calculate_tf(inverted_index,passages_id_and_terms_info)

# calculate the tf-idf for each term in each passage/query
def calculate_passage_or_query_tf_idf(term_tf_in_passage_or_query, idf, passages_or_queries_id_and_terms_info):
    tf_idf_for_passage_or_query = defaultdict(dict)
    for pid, terms in passages_or_queries_id_and_terms_info.items():
        tf_idf_cur_passage_or_query = {}
        for term in terms:
            # Each word will be counted only once
            if term not in tf_idf_cur_passage_or_query:
                tf_idf_cur_passage_or_query[term] = term_tf_in_passage_or_query[term][pid] * idf[term]

        tf_idf_for_passage_or_query[pid] =  tf_idf_cur_passage_or_query      
    return tf_idf_for_passage_or_query

tf_idf_for_passage =  calculate_passage_or_query_tf_idf(term_tf_in_passage, idf, passages_id_and_terms_info)

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
            # If there is a corresponding word it is stored, if no corresponding word is found additional processing is performed
            query_contain_token = [token.upper() for token in tokens if token.upper() in vocabulary_from_task1]
            if query_contain_token:
                queries_id_and_terms_info[qid] = query_contain_token
            query_contain_token = [token.capitalize() for token in tokens if token.capitalize() in vocabulary_from_task1]
            if query_contain_token:    
                queries_id_and_terms_info[qid] = query_contain_token
            if qid not in queries_id_and_terms_info:        
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

# calculate the tf for each term in each query
term_tf_in_query = calculate_tf(inverted_index_query,queries_id_and_terms_info)

# calculate the tf-idf for each term in each query
tf_idf_for_query =  calculate_passage_or_query_tf_idf(term_tf_in_query, idf, queries_id_and_terms_info)

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

# ===================================================BM25======================================================================
k1, k2, b = 1.2, 100, 0.75
dl = passages_id_and_terms_info
avdl = 0
for total_words_in_passage in dl.values():
    avdl += len(total_words_in_passage)
avdl = avdl / len(dl)

def calculate_BM25(n, f, qf, R, r, N, dl, avdl, k1, k2, b, pid):
    K = k1 * ((1 - b) + (b * (len(dl[pid])/avdl)))
    BM25_part1 = np.log( ((r+0.5) / (R-r+0.5)) / ((n-r+0.5) / (N-n-R+r+0.5)) )
    BM25_part2 = ((k1+1) * f) / (K + f)
    BM25_part3 = ((k2 + 1) * qf) / (k2 + qf)
    return BM25_part1 * BM25_part2 * BM25_part3

# n = len(inverted_index[term])
# f = inverted_index[term][pid] 
def calculate_BM25_score(qid_and_pid, queries_id_and_terms_info, inverted_index, inverted_index_query, dl, avdl, k1, k2, b):
    BM25_score = defaultdict(dict)
    for qid, pids in qid_and_pid.items():
        for pid in pids:
            BM25_cur_qid_score = 0
            for term in queries_id_and_terms_info[qid]:
                BM25_cur_qid_score += calculate_BM25(len(inverted_index[term]), inverted_index[term].get(pid, 0), inverted_index_query[term][qid], 0, 0, len(dl), dl, avdl, k1, k2, b, pid)
            if pid not in BM25_score[qid]:
                BM25_score[qid][pid] = BM25_cur_qid_score
    return BM25_score

BM25_score = calculate_BM25_score(qid_and_pid, queries_id_and_terms_info, inverted_index, inverted_index_query, dl, avdl, k1, k2, b)

# select the top 100
def BM25_top100(BM25_score, queries_id_and_terms_info):
    top_100_pids_for_qid = defaultdict(list)
    for qid, pid_scores in BM25_score.items():
        sorted_pids = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
        top_100_pids = sorted_pids[:100]
        top_100_pids_for_qid[qid] = top_100_pids 
    with open('bm25.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for qid, _ in queries_id_and_terms_info.items():
            top100_passage = top_100_pids_for_qid[qid]
            for pid,score in top100_passage:
                writer.writerow([qid, pid, score])

BM25_top100(BM25_score, queries_id_and_terms_info)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"task3 running time：{elapsed_time} second")
