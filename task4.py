from collections import defaultdict
import json
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

with open('queries_id_and_terms_info.json', 'r', encoding='utf-8') as file:
    queries_id_and_terms_info = json.load(file) 

with open('passages_id_and_terms_info.json', 'r', encoding='utf-8') as file:
    passages_id_and_terms_info = json.load(file)                    



end_time = time.time()
elapsed_time = end_time - start_time
print(f"task3程序运行时间：{elapsed_time}秒")    