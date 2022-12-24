# Part 2 : Truth to be told

- In this problem, we are using naive baye's rule to classify whether the user generated reviwe is deceptive or truthful.
- The classifier dunction takes two arguments(train data and test data which are both dictionaries)
- We are using bayes rule to calculate the posterior probability of the review. Which means probability of it being deceptive given the words in the review.
- P(deceptive|word1.word2.word3....wordn) = P(word1.word2.word3....wordn|deceptive)*P(deceptive)/P(word1.word2.word3....wordn)
- P(truthful|word1.word2.word3....wordn) = P(word1.word2.word3....wordn|truthful)*P(truthful)/P(word1.word2.word3....wordn)
- Here, the P(deceptive) is the number of deceptive reviews/total reviews and same logic for P(truthful)
- We first need to clean the train data sentences as they contain punctuations and are in string.
- The clean_sentence functions is used to clean the data. It removes the punctuation, splits the string into list and removes the stop words.
- After the data is cleaned , we then calculate the probabilities of P(word1|deceptive), P(word2|deceptive),etc. In this, we have used laplace smoothning to handle cases where word does not exist in the probability dicionary. 
- Ex: prob_deceptive[word] = (n_word_dec + alpha) / (deceptive_sent_count + alpha*len(vocab)). Here alpha is set to 1 for laplace smoothning. 
- Once we calculate the P(word|deceptive) and P(word|truthful), then we iterate over the test set, clean the sentences and find the posterior probabilities. 
- if deceptive_posterior  > truthful_posterior then we mark the sentence as deceptive and vice versa. We append these labels to a list and return.
- References: https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece -- laplace smoothning 
- https://github.com/stopwords-iso/stopwords-en/blob/master/stopwords-en.txt : stop words

