# MC_Upsampling
*Work in progress*

This project attempts an upsampling solution for imbalanced classification tasks using a Markov Chain for text generation (nicknamed "MCUpsampling"). Other solutions tested in this project include bootstrapped upsampling of the minority class, bootstrapped downsampling of the majority class, and using randomly generated bags of words. These solutions are tested by classifying documents of a corpus as either positive (1) or negative (0). There are 1/3 as many positively labeled documents in the corpus as there are negatively labeled documents. The documents are fed through a pipeline consisting of Sci-Kit Learn's Tfidf Transformer and Multinomial Naive-Bayes classifier.

Results so far have demonstrated that MCUpsampling can increase accuracy by up to 8 additional percentage points and recall by up to 70 additional percentage points. MCUpsampling performs noteably better than boostrapped downsampling and random bags of words, but only slightly better than bootstrapped upsampling. 

---

Imbalanced      0.785   0.1485148514851485

MCUpsampling    0.8625  0.7821782178217822

Bootstrap up    0.85    0.7722772277227723

Bootstrap down  0.8425  0.4158415841584158

Random bag      0.825   0.8118811881188119
