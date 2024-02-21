import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
from collections import Counter

# nltk.download('punkt')
# nltk.download('stopwords')

def preprocessing(text):
    # handle the separators
    text = text.replace("/", " ")
    # group tokens with minor differences
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenisation
    text = nltk.word_tokenize(text)
    return text

with open('passage-collection.txt', 'r', encoding='utf-8') as file:
    text = file.read()
  
terms = preprocessing(text) 
number_of_terms = Counter(terms)

total_count_of_terms = sum(number_of_terms.values())
sorted_number_of_terms = number_of_terms.most_common()

normalized_frequencies = []
for term, count in sorted_number_of_terms:
    normalized_frequency = count / total_count_of_terms
    normalized_frequencies.append(normalized_frequency)

N = len(sorted_number_of_terms)
frequency_ranking = range(1, N + 1)

plt.figure(figsize=(10, 6))
plt.plot(frequency_ranking, normalized_frequencies, linestyle='-', label='data')
plt.title('Probability of Occurrence against Frequency Ranking')
plt.xlabel('Frequency ranking(log)')
plt.ylabel('Term probability of occurrence(log)')
plt.xlim(left=1)
plt.ylim(bottom=(10 ** -7))
plt.legend()

s = 1
HN = np.sum([1 / (n ** s) for n in range(1, N + 1)])
Zipf_law_distribution = [(1 / (k ** s)) / HN for k in range(1, N + 1)]

plt.figure(figsize=(10, 6))
plt.loglog(frequency_ranking, normalized_frequencies, linestyle='-', label='empirical distribution')
plt.loglog(frequency_ranking, Zipf_law_distribution, linestyle='--', color='red', label='actual Zipf’s law distribution')
plt.title(' Empirical distribution compare to the actual Zipf’s law distribution')
plt.xlabel('Frequency rankin(log)')
plt.ylabel('Term probability of occurrence(log)')
plt.xlim(left=1)
plt.ylim(bottom=(10 ** -7))
plt.legend()

plt.show()

