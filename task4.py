from collections import Counter, defaultdict
import json
import csv
import time
import numpy as np

start_time = time.time()

# read the tsv to get the connection between qid and pid
qid_and_pid = defaultdict(set)
with open('candidate-passages-top1000.tsv', 'r', encoding='utf-8') as file:
    tsv_reader = csv.reader(file, delimiter='\t')
    for row in tsv_reader:
        qid,pid = row[0],row[1]
        if qid not in qid_and_pid[qid]:
            qid_and_pid[qid].add(pid)    

with open('inverted_index.json', 'r', encoding='utf-8') as file:
    inverted_index = json.load(file)
    
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

def calcaulate_laplace(queries_id_and_terms_info, inverted_index, passages_id_and_terms_info, qid_and_pid):
    #  qid :{pid:score}
    laplace_estimates_scores =  defaultdict(dict)
    for qid, pids in qid_and_pid.items():
        query_words = queries_id_and_terms_info[qid]
        for pid in pids:
            # word : score
            temp_word_score = {}
            laplace_estimates_scores_pid = 0
            for word in query_words:
                # query - passage - score
                probabilities_of_cur_word = np.log((inverted_index[word].get(pid,0) + 1) / (len(passages_id_and_terms_info[pid]) + len(set(passages_id_and_terms_info[pid]))))
                if word not in temp_word_score:
                    temp_word_score[word] = probabilities_of_cur_word
            for word_score in temp_word_score.values():
                    laplace_estimates_scores_pid += word_score           
            if pid not in laplace_estimates_scores[qid]:
                laplace_estimates_scores[qid][pid] = laplace_estimates_scores_pid  
    return laplace_estimates_scores                       

laplace_estimates_scores = calcaulate_laplace(queries_id_and_terms_info, inverted_index, passages_id_and_terms_info,qid_and_pid)

def top100(scores, filname):
    top_100_pids_for_qid = defaultdict(list)
    for qid, pid_scores in scores.items():
        sorted_pids = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
        top_100_pids = sorted_pids[:100]
        top_100_pids_for_qid[qid] = top_100_pids 
    with open(filname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for qid, _ in queries_id_and_terms_info.items():
            top100_passage = top_100_pids_for_qid[qid]
            for pid,score in top100_passage:
                writer.writerow([qid, pid, score])

top100(laplace_estimates_scores, 'laplace.csv') 

def calculate_lidstone(queries_id_and_terms_info, inverted_index, passages_id_and_terms_info, qid_and_pid, e):
    lidstone_scores =  defaultdict(dict)
    for qid, pids in qid_and_pid.items():
        query_words = queries_id_and_terms_info[qid]
        for pid in pids:
            # word : score
            temp_word_score = {}
            lidstone_scores_pid = 0
            for word in query_words:
                # query - passage - score
                probabilities_of_cur_word = np.log((inverted_index[word].get(pid,0) + e) / (len(passages_id_and_terms_info[pid]) + len(set(passages_id_and_terms_info[pid])) * e))
                # probabilities_of_cur_word = (inverted_index[word].get(pid,0) + e) / (len(passages_id_and_terms_info[pid]) + len(Counter(passages_id_and_terms_info[pid])) * e)
                if word not in temp_word_score:
                    temp_word_score[word] = probabilities_of_cur_word
            for word_score in temp_word_score.values():
                    lidstone_scores_pid += word_score           
            if pid not in lidstone_scores[qid]:
                lidstone_scores[qid][pid] = lidstone_scores_pid  
    return lidstone_scores

lidstone_scores = calculate_lidstone(queries_id_and_terms_info, inverted_index, passages_id_and_terms_info, qid_and_pid, e=0.1)
l1 = calculate_lidstone(queries_id_and_terms_info, inverted_index, passages_id_and_terms_info, qid_and_pid, e=0.4)
l2 = calculate_lidstone(queries_id_and_terms_info, inverted_index, passages_id_and_terms_info, qid_and_pid, e=0.7)

top100(lidstone_scores, 'lidstone.csv')
top100(l1, 'lidstone1.csv')
top100(l2, 'lidstone2.csv')

def calculate_dirichlet(queries_id_and_terms_info, inverted_index, vocabulary_size, passages_id_and_terms_info, qid_and_pid, u):
    dirichlet_scores =  defaultdict(dict)
    for qid, pids in qid_and_pid.items():
        query_words = queries_id_and_terms_info[qid]
        for pid in pids:
            # word : score
            temp_word_score = {}
            dirichlet_scores_pid = 0
            for word in query_words:
                # query - passage - score
                passage_len = len(passages_id_and_terms_info[pid])
                tf_in_vocabulary = 0
                for tf in inverted_index[word].values():
                    tf_in_vocabulary += tf 
                probabilities_of_cur_word = np.log((passage_len / (passage_len + u)) * (inverted_index[word].get(pid,0) / passage_len) + ((u / (passage_len + u)) * (tf_in_vocabulary / vocabulary_size)))
                if word not in temp_word_score:
                    temp_word_score[word] = probabilities_of_cur_word
            for word_score in temp_word_score.values():
                    dirichlet_scores_pid += word_score           
            if pid not in dirichlet_scores[qid]:
                dirichlet_scores[qid][pid] = dirichlet_scores_pid            
    return dirichlet_scores

vocabulary_size = 0

for term in inverted_index:
    for tf in inverted_index[term].values():
        vocabulary_size += tf

dirichlet_scores = calculate_dirichlet(queries_id_and_terms_info, inverted_index, vocabulary_size, passages_id_and_terms_info, qid_and_pid, u=50)

top100(dirichlet_scores, 'dirichlet.csv')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"task4 running time：{elapsed_time} second")  

# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取CSV文件
# df1 = pd.read_csv('lidstone.csv')
# df2 = pd.read_csv('laplace.csv')
# df3 = pd.read_csv('dirichlet.csv')

# column1 = df1.iloc[:100, 1]
# column2 = df2.iloc[:100, 1]
# column3 = df3.iloc[:100, 1]

# # 使用集合找出两列中不同的值，并计算数量
# same_values = 0
# same_values_1 = 0
# same_values_2 = 0
# for i in column1:
#     for j in column2:
#         if i == j:
#             same_values+=1

# for i in column1:
#     for j in column3:
#         if i == j:
#             same_values_1 += 1    

# for i in column2:
#     for j in column3:
#         if i == j:
#             same_values_2 += 1                  

# print(same_values)
# print(same_values_1)
# # 创建条形图来可视化不同值的数量
# plt.figure(figsize=(8, 6)) # 设置图形大小
# plt.bar(['e=0.4 compare to e=0.1','e=0.7 compare to e=0.1'], [100-same_values, 100 - same_values_1])
# plt.title('Number of Different Values of the 100 Rows')
# plt.ylabel('Count')
# plt.savefig('different.pdf')
# plt.show()

    

    