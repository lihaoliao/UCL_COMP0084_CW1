import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
import time

# nltk.download('punkt')
# nltk.download('stopwords')

def preprocessing(text, remove):
    # handle the separators
    text = text.replace("/", " ")
    # group tokens with minor differences
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenisation
    tokens = nltk.word_tokenize(text)
    unremove_text = Counter(tokens)
    if remove:
        stop_words = set(stopwords.words('english'))
        capitalized_stop_words = {word.capitalize() for word in stop_words}
        stop_words.update(capitalized_stop_words)
        remove_tokens = [word for word in tokens if word not in stop_words]
        remove_text = Counter(remove_tokens)
    
    return unremove_text, remove_text


start_time = time.time()
with open('passage-collection.txt', 'r', encoding='utf-8') as file:
    text = file.read()
   
unremove_number_of_terms, remove_number_of_terms = preprocessing(text, True)
print(len(unremove_number_of_terms))
print(len(remove_number_of_terms)) 

unremove_total_count_of_terms = sum(unremove_number_of_terms.values())
unremove_sorted_number_of_terms = unremove_number_of_terms.most_common()

unremove_normalized_frequencies = []
for term, count in unremove_sorted_number_of_terms:
    normalized_frequency = count / unremove_total_count_of_terms
    unremove_normalized_frequencies.append(normalized_frequency)

N = len(unremove_sorted_number_of_terms)
unremove_frequency_ranking = range(1, N + 1)
s = 1
HN = np.sum([1 / (n ** s) for n in range(1, N + 1)])
unremove_Zipf_law_distribution = [(1 / (k ** s)) / HN for k in range(1, N + 1)]

plt.figure(figsize=(10, 6))
plt.plot(unremove_frequency_ranking, unremove_normalized_frequencies, linestyle='-', label='data')
plt.plot(unremove_frequency_ranking, unremove_Zipf_law_distribution, linestyle='--', color='red', label='theory (Zipf\'s law)')
plt.title('Probability of Occurrence against Frequency Ranking')
plt.xlabel('Frequency ranking')
plt.ylabel('Term probability of occurrence')
plt.legend()
# plt.savefig('figure1.pdf')

plt.figure(figsize=(10, 6))
plt.loglog(unremove_frequency_ranking, unremove_normalized_frequencies, linestyle='-', label='data')
plt.loglog(unremove_frequency_ranking, unremove_Zipf_law_distribution, linestyle='--', color='red', label='theory (Zipf\'s law)')
plt.title(' Empirical distribution compare to the actual Zipf’s law distribution')
plt.xlabel('Frequency rankin(log)')
plt.ylabel('Term probability of occurrence(log)')
plt.xlim(left=1)
plt.legend()
# plt.savefig('figure2.pdf')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序运行时间：{elapsed_time}秒")
plt.show()

