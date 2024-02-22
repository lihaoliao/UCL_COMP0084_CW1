import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from scipy.special import rel_entr
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


# start_time = time.time()
with open('passage-collection.txt', 'r', encoding='utf-8') as file:
    text = file.read()
   
unremove_number_of_terms, remove_number_of_terms = preprocessing(text, True)

unremove_total_count_of_terms = sum(unremove_number_of_terms.values())
remove_total_count_of_terms  = sum(remove_number_of_terms.values())

unremove_sorted_number_of_terms = unremove_number_of_terms.most_common()
remove_sorted_number_of_terms = remove_number_of_terms.most_common()


def calculate_normalized_frequencies(sorted_number_of_terms, total_count_of_terms):
    return [count / total_count_of_terms for term, count in sorted_number_of_terms]

unremove_normalized_frequencies = calculate_normalized_frequencies(unremove_sorted_number_of_terms, unremove_total_count_of_terms)
remove_normalized_frequencies = calculate_normalized_frequencies(remove_sorted_number_of_terms, remove_total_count_of_terms)

s = 1

unremove_N = len(unremove_sorted_number_of_terms)
remove_N = len(remove_sorted_number_of_terms)

unremove_frequency_ranking = range(1, unremove_N + 1)
remove_frequency_ranking = range(1, remove_N + 1)

unremove_HN = np.sum([1 / (n ** s) for n in range(1, unremove_N + 1)])
remove_HN = np.sum([1 / (n ** s) for n in range(1, remove_N + 1)])

unremove_Zipf_law_distribution = [(1 / (k ** s)) / unremove_HN for k in range(1, unremove_N + 1)]
remove_Zipf_law_distribution = [(1 / (k ** s)) / remove_HN for k in range(1, remove_N + 1)]

plt.figure(figsize=(10, 6))
plt.plot(unremove_frequency_ranking, unremove_normalized_frequencies, linestyle='-', label='data')
plt.plot(unremove_frequency_ranking, unremove_Zipf_law_distribution, linestyle='--', color='red', label='theory (Zipf\'s law)')
plt.title('Probability of Occurrence against Frequency Ranking')
plt.xlabel('Frequency ranking')
plt.ylabel('Term probability of occurrence')
plt.legend()
plt.savefig('figure1.pdf')

plt.figure(figsize=(10, 6))
plt.loglog(unremove_frequency_ranking, unremove_normalized_frequencies, linestyle='-', label='data')
plt.loglog(unremove_frequency_ranking, unremove_Zipf_law_distribution, linestyle='--', color='red', label='theory (Zipf\'s law)')
plt.title(' Empirical distribution compare to the actual Zipf’s law distribution')
plt.xlabel('Frequency rankin(log)')
plt.ylabel('Term probability of occurrence(log)')
plt.xlim(left=1)
plt.legend()
# plt.savefig('figure2.pdf')

plt.figure(figsize=(10, 6))
plt.loglog(remove_frequency_ranking, remove_normalized_frequencies, linestyle='-', label='data')
plt.loglog(remove_frequency_ranking, remove_Zipf_law_distribution, linestyle='--', color='red', label='theory (Zipf\'s law)')
plt.title(' Empirical distribution compare to the actual Zipf’s law distribution without stop words')
plt.xlabel('Frequency rankin(log)')
plt.ylabel('Term probability of occurrence(log)')
plt.xlim(left=1)
plt.legend()
# plt.savefig('figure3.pdf')

# Kullback–Leibler divergence calculate and guarantee sum is 1
# unremove_normalized_frequencies_total = np.array(unremove_normalized_frequencies).sum()
# unremove_normalized_frequencies /= unremove_normalized_frequencies_total
# unremove_Zipf_law_distribution_total = np.array(unremove_Zipf_law_distribution).sum()
# unremove_Zipf_law_distribution /= unremove_Zipf_law_distribution_total
# Kullback_Leibler_divergence_unremoval = rel_entr(unremove_normalized_frequencies, unremove_Zipf_law_distribution)
# Kullback_Leibler_divergence_unremoval = np.sum(Kullback_Leibler_divergence_unremoval)

# remove_normalized_frequencies_total = np.array(remove_normalized_frequencies).sum()
# remove_normalized_frequencies /= remove_normalized_frequencies_total
# remove_Zipf_law_distribution_total = np.array(remove_Zipf_law_distribution).sum()
# remove_Zipf_law_distribution /= remove_Zipf_law_distribution_total
# Kullback_Leibler_divergence_removal = rel_entr(remove_normalized_frequencies, remove_Zipf_law_distribution)
# Kullback_Leibler_divergence_removal = np.sum(Kullback_Leibler_divergence_removal)

# plt.figure(figsize=(8, 6))
# plt.bar(['Remove_stop_words', 'Unremove_stop_words'], [Kullback_Leibler_divergence_removal, Kullback_Leibler_divergence_unremoval])
# plt.title('Kullback Leibler Divergence Comparison')
# plt.xlabel('Condition')
# plt.ylabel('Kullback Leibler Divergence')
# plt.savefig('Kullback_Leibler_Divergence.pdf')

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"程序运行时间：{elapsed_time}秒")
plt.show()

