from collections import defaultdict
import os
import json
import math

# 这个是tf的分子,就是每个term在某个passage出现的次数
with open('inverted_index.json', 'r', encoding='utf-8') as file:
    inverted_index = json.load(file)

# 这个变量长度是idf分子 然后它每个passage的value是tf的分母
with open('each_passage_terms_sum.json', 'r', encoding='utf-8') as file:
    each_passage_terms_sum = json.load(file)


total_passage = len(each_passage_terms_sum)

# calculate IDF for each term - shared by query and passage
idf = {term: math.log10(total_passage/len(inverted_index[term])) for term in inverted_index}

# calculate the TF for each term in each passage
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

# calculate the tf-idf for each passage

# calculate the idf for each term in each query

