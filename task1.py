import matplotlib.pyplot as plt
import numpy as np
# import nltk
import re
from collections import Counter
# from nltk.corpus import stopwords
from scipy.special import rel_entr
import time

# nltk.download('stopwords')
preprocessing_re = re.compile(r'[^a-zA-Z\s]')

def preprocessing(text, remove):
    # handle the separators
    text = text.replace("/", " ")
    # group tokens with minor differences
    text = re.sub(preprocessing_re, '', text)
    # Tokenisation
    tokens = text.split()
    unremove_text = Counter(text.split())
    remove_text = Counter()
    if remove:
        stop_words = {'Has', 'Doesn', 'ma', 'them', 'You', 'When', "Should've", 'After', "shouldn't", 'as', "isn't", 'On', 'hadn', 'were', "she's", 'no', 'out', 'Mustn', 'Mightn', 'mightn', 'didn', 'Here', 'what', 'has', 'Again', 'very', 'Where', 'Ain', 'Myself', 've', 'Been', 'We', 'my', 'And', 'won', 'Each', 'those', 't', 'again', "you'd", 'Down', 'doing', 'down', 'Can', 'Below', 'than', "doesn't", 'when', 'Than', 'Ve', "you've", 'y', 'D', 'does', "Isn't", 'M', 'But', 'from', 'weren', 'Nor', 'Weren', 'myself', 'Was', 'and', 'above', 'o', 'To', "wouldn't", 'few', 'An', 'By', 'Both', 'why', 'Shan', 'Am', 'did', 'this', 'both', 'they', 'd', 'me', 'O', 'our', 'shouldn', 'we', 'whom', "mightn't", 'more', "weren't", "mustn't", 'against', 'Until', 'mustn', 'then', 'Couldn', 'Itself', 'How', 'be', 'off', 'once','so', 'Hers', 'don', 'Very', 'Be', 'Out', 'in', 'Were', 's', 'Shouldn', 'he', "Wouldn't", 'while', 'Aren', "Weren't", 'wasn', 'Re', 'Into', 'having', 'can', 'itself', "it's", 'not', 'shan', 'Themselves', 'theirs', "You'd", 'Ll', 'Those', 
        'his', 'before', 'during', 'My', "You've", 'Had', 'Do', 'Only', "you'll", 'Ma', 'All', "Aren't", 'Yourself', "needn't", "Mightn't", 'I', 'Too', 'there', 'Isn', 'i', 'yourself', "Doesn't", 'she', 'it', "Didn't", 'Further', "Hadn't", 'you', 'ain', "Won't", 'Haven', 'here', 'on', 'The', 'Just', 'into', 'after', 're', 'up', 'Needn', 'Have', 'While', 'll', 'That', 'same', 'should', 'Wouldn', 'over', 'under', 'Doing', 'do', 'their', 'of', 'further', "aren't", 'isn', 'Because', 'hers', 'are', 'for', 'Through', "hadn't", 'but', 'Such', 'During', 'ourselves', 'which', 'His', 'aren', 'until', "Don't", 'such', 'She', 'Over', 'had', 'will', "Haven't", 'Don', 'S', 'Before', 'Ourselves', 'these', "couldn't", "haven't", 'Own', 'own', 'him', 'Above', 'Same', 'at', 'most', 'himself', 'Himself', 'Did', 'now', 'Being', 'that', 'have', 'About', 'Up', 'only', 'Hadn', 'Its', 'Him', 'Didn', 'through', 'Should', 'is', "won't", 'if', "You're", 'Some', 'yours', 'Won', 'some', 'Them', 'with', 'Who', 'Once', "She's", 'Most', 'Ours', 'Herself', "that'll", "Shouldn't", 'No', 'wouldn', 'nor', 'between', 'From', "You'll", 'They', 'needn', 'other', 'Your', 'hasn', 'Any', 'Yourselves', 'any', 'At', 'Does', 'More', 'In', 'its', 'Or', 'a', 'With', 'Few', "shan't", 'These', 'So', 'Against', 'This', 'couldn', 'Her', 'all', 'too', "hasn't", 'Of', 'Having', "didn't", 'There', 'just', 'Wasn', 'Are', "Needn't", 'or', 'Whom', 'It', "should've", 'T', 'am', 'Y', 'For', 'where', 'about', 'Under', 'If', 'each', "It's", 'your', 'Then', 'What', 'who', 'Off', 'Theirs', 'Our', 'Is', 'because', 'to', 'an', 'Other', 'Me', 'haven', 'themselves', 'Not', 'Now', "Wasn't", "don't", 'Their', 'below', 'her', 'ours', 'being', "Couldn't", 'doesn', 'was', 'been', "Shan't", 'He', "you're", 'Which', 'Between', "wasn't", "That'll", 'by', 'Yours', 'Hasn', 'Will', 'herself', 'Why', 'As', "Mustn't", 'A', "Hasn't", 'how', 'the', 'yourselves', 'm'}
        remove_text = Counter(word for word in tokens if word not in stop_words)
        with open('remove_stop_word_vocabulary.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(remove_text.keys()))
    return unremove_text, remove_text

# def calculate_proportion_in_range(normalized_frequencies, N, a, b):
#     count = sum(1 for freq in normalized_frequencies if a <= freq <= b)
#     proportion = count / N
#     return proportion


start_time = time.time()
with open('passage-collection.txt', 'r', encoding='utf-8') as file:
    text = file.read()
   
unremove_number_of_terms, remove_number_of_terms = preprocessing(text, True)

unremove_total_count_of_terms = sum(unremove_number_of_terms.values())
remove_total_count_of_terms  = sum(remove_number_of_terms.values())

unremove_sorted_number_of_terms = unremove_number_of_terms.most_common()
remove_sorted_number_of_terms = remove_number_of_terms.most_common()

# depends on the count (term frequency)
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

# k - rank s = 1  HN - related ti N - vocabulary rank
unremove_Zipf_law_distribution = [(1 / (k ** s)) / unremove_HN for k in range(1, unremove_N + 1)]
remove_Zipf_law_distribution = [(1 / (k ** s)) / remove_HN for k in range(1, remove_N + 1)]

# ===========================Proportion of terms within a certain range================================
# a = 10**-5
# b = 10**-2.5
# proportion = calculate_proportion_in_range(remove_normalized_frequencies, remove_N, a, b)
# proportion_Z = calculate_proportion_in_range(remove_Zipf_law_distribution, remove_N, a, b)
# plt.figure(figsize=(8, 6))
# conditions = ['normalized_frequencies', 'Zipf_law_distribution']
# proportions = [proportion*100, proportion_Z*100]
# bars = plt.bar(conditions, proportions)
# for bar, proportion in zip(bars, proportions):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, height, f'{proportion:.2f}%', ha='center', va='bottom')

# plt.title('Terms proportion between $10^{-2.5}$ and $10^{-5}$')
# plt.xlabel('Condition')
# plt.ylabel('Proportion')
# plt.savefig('Terms_proportion.pdf')
# ===========================Proportion of terms within a certain range================================

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

plt.figure(figsize=(10, 6))
plt.loglog(remove_frequency_ranking, remove_normalized_frequencies, linestyle='-', label='data')
plt.loglog(remove_frequency_ranking, remove_Zipf_law_distribution, linestyle='--', color='red', label='theory (Zipf\'s law)')
plt.title(' Empirical distribution compare to the actual Zipf’s law distribution without stop words')
plt.xlabel('Frequency rankin(log)')
plt.ylabel('Term probability of occurrence(log)')
plt.xlim(left=1)
plt.legend()
# plt.savefig('figure3.pdf')

# =====================================Kullback–Leibler divergence============================================
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
# conditions = ['Remove_stop_words', 'Unremove_stop_words']
# proportions = [Kullback_Leibler_divergence_removal, Kullback_Leibler_divergence_unremoval]
# bars = plt.bar(conditions, proportions)
# for bar, proportion in zip(bars, proportions):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, height, f'{proportion:.2f}', ha='center', va='bottom')
# plt.title('Kullback Leibler Divergence Comparison')
# plt.xlabel('Condition')
# plt.ylabel('Kullback Leibler Divergence')
# plt.savefig('Kullback_Leibler_Divergence.pdf')
# =====================================Kullback–Leibler divergence============================================

end_time = time.time()
elapsed_time = end_time - start_time
print(f"task1 running time：{elapsed_time} second")
plt.show()

