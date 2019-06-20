# Monde Sinxi - Skills Test Report

## Setup

`python -m venv env`

`pip install -r requirements.txt`


## Task 1: Classifying the document


### Methodology

We use three methods for genrating vectors to feed into the models

#### Data Preparation

1. Remove symbols, numbers, extra spaces and puctuation. Make all text lower case.
2. Lemantize all words to reduce all words to only their roots.
3. Remove all stop words.

#### Tokenization

* Bag-of-words: Use `CountVectorizer` from `sklearn.feature_extraction.text`, which performs a raw count of the number of tokens are present in the corpus. Each document is represented by a sparse vector
  
* TF-IDF: Use `TfidVectorizer` from `sklearn.feature_extraction.text`, a combination of `CountVectorizer` and `TfidTransform`. Also performs vectorization of text, however, any words repeated across documents have a reduced contribution to the vector, giving more unique tokens a higher degree of representation in the document vector.
  
* Word embedding (word2vec): Each word is converted to a vector. Conversion could be trained from the available text or use a pre-trained model could be used to assign vectors. For the entire document, the average vector is obtained to feed into the models. For this report we use a pre-trained model from `Glov` that was trained from 2 billion tweets.

#### Models

* Naive Bayes
* SGD linear
* Logistic Regression
* Decision Tree
* Randon Forrest
* Neural Network
* Neural network with word embedding


### Results

#### F1-score matrix

|  | Naive-Bayes | SGD (linear SVM) | Logistic Regression | Decision Tree | Random Forrest | Neural Network | Deep Neural Network| 
| ----- | -----  | ----- | ------ | ------ | ------ | ------ | ------ |
| Bag-of-words |  | | | | | | |
| TF-IDF | | | | | | | |
| word2vec | | | | | | | |

#### Plots of confusion matrices

[]()

#### Selecting the best model/tokenization

## Task 2: Data Exploration and Insights

### Frequent Terms

#### Unigram

#### Bigram

#### Trigram

### Sentiment Analysis

The `Vader` library wrapped in `sklearn` is used to perform a sentiment analysis. There exists a lexicon in `vader` that coumpunds the individual contributions of words to determine the components that convey sentiment i.e positive, negative and neutral. 

An example

### Topic Extraction

The `LatentDirichletAllocation` module from `sklearn.decomposition` was used to extract the underlying topics from text. 

An example

### Unsupervised Learning - Clustering Analysis

From `sklearn.cluster` we use the `KMeans` module to perorm the clustering analysis. 

## Useful Resources
