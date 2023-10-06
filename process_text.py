import re
import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import defaultdict
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def clean_text(text):
    # Remove non-alphanumeric characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert multiple whitespace characters to a single space
    text = re.sub(r'\s+', ' ', text)
    # Convert the text to lowercase
    text = text.lower()
    return text

def tokenize_document(document):
    tokens = word_tokenize(document)
    return tokens

def filter_nouns(tokens):
    # load the English language model
    nlp = spacy.load("en_core_web_sm")
    
    # join the tokens into a single text string
    text = ' '.join(tokens)
    
    # process the text with spaCy
    doc = nlp(text)
    
    # extract nouns ('NN' tags)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    
    return nouns

def get_most_frequent_words(corpus, dictionary):
    # Initialize a defaultdict to hold the frequency of each word
    frequency = defaultdict(int)
    
    # Iterate over each document in the corpus
    for doc in corpus:
        # Iterate over each word and its count in the document
        for word_id, count in doc:
            # Add the count to the word's frequency
            frequency[dictionary[word_id]] += count
    
    # Sort the words by frequency in descending order and return them
    sorted_words = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words

def lemmatize_doc(document):
    lemmatizer = WordNetLemmatizer()
    document_lemmatized = []
    for word in document:
        document_lemmatized.append(lemmatizer.lemmatize(word))

    return document_lemmatized

def remove_stopwords(token_docs, WORDS_FILTER = []):
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    documents_no_stop = []
    for doc in tqdm(token_docs): # each doc is a list of words
        # loop through each word and filter
        words_filter = [word for word in doc if word not in stop_words and word not in WORDS_FILTER] 

        # append to new list of lists
        documents_no_stop.append(words_filter)
    
    return documents_no_stop


def process_amazon_reviews_file(input_file, output_file):
  
  # read file
  df = pd.read_csv(input_file)

  # clean text
  df['Comment'] = df['Comment'].apply(clean_text)
  df['Reviews'] = df['Reviews'].apply(clean_text)

  # remove 'read more' from reviews
  df['Reviews'] = df['Reviews'].apply(lambda x: x.replace('read more', ''))

  # tokenize
  token_docs = []
  for doc in df['Reviews']:
      token_docs.append(tokenize_document(doc))
  print(token_docs[0])

  # remove stopwords, filter words
  nltk.download('stopwords')

  documents_no_stop = remove_stopwords(token_docs)

  print(documents_no_stop[0])

  # lemmatize
  docs_lemmatized = []

  for doc in documents_no_stop:
      docs_lemmatized.append(lemmatize_doc(doc))

  print(docs_lemmatized[0])

  # export treated and lemmatized docs to parquet file format
  df_lemmatized = pd.DataFrame(columns=["Ratings","Comment","Review_tokens"])
  df_lemmatized["Ratings"] = df["Ratings"]
  df_lemmatized["Comment"] = df["Comment"]
  df_lemmatized["Review_tokens"] = pd.Series(docs_lemmatized)

  df_lemmatized.to_parquet(output_file)

  return df_lemmatized

if __name__ == "__main__":
    df_lemmatized = process_amazon_reviews_file('data/APPLE_iPhone_SE.csv', 'data/amazon_reviews.parquet')
    print(len(df_lemmatized))