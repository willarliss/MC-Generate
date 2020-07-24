# MC_Upsampling
*Work in progress*

This project attempts an upsampling solution for imbalanced classification tasks using a Markov Chain for text generation (MCGen). Other solutions tested in this project include bootstrapped upsampling of the minority class, bootstrapped downsampling of the majority class, and randomly generated bags of words. These solutions are tested by attempting to classify documents in unbalanced corpora. Testing has been performed on both a Linear SVC and a Multinomial Naive-Bayes classifier using TF-IDF transformation. 
 - The MCGen model has been updated to include a combination of first-order and (mostly) second-order chains. A minimum document length has also been introduced.

### Multinomial Naive-Bayes
Testing on a binomial corpus, results so far have demonstrated that MCGen can increase accuracy by up to 7-10 additional percentage points, ROC0-AOC by up to 25 additional percentage points, and recall by up to 65 additional percentage points. MCGen performs noteably better than boostrapped downsampling and random bags of words, but only slightly better than bootstrapped upsampling.

### Linear SVC
Testing on a binomial corpus, results so far have demonstrated greater performance across the board than Naive-Bayes. MCGen's performance, however, is only marginally better than using an imbalanced corpus in terms of accuracy and ROC-AUC, but up to 10 percentage points greater in recall. The increase (and low variability) in performance across class balancing models is likely testiment to Linear SVC's strong ability to classify text.
