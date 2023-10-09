import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lib.process_text import clean_text, tokenize_document, lemmatize_doc
from nltk.corpus import stopwords, words
import torch
import torch.nn.functional as F
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Load the tokenizer and model
model = AutoModelForSequenceClassification.from_pretrained("data/review_classifier")
tokenizer = AutoTokenizer.from_pretrained("data/review_classifier_tokenizer")

# download nltk data
nltk.data.path = [os.path.join(os.getcwd(), "data/nltk_data")]
nltk.download('vader_lexicon',download_dir=nltk.data.path[0])
nltk.download('punkt',download_dir=nltk.data.path[0])
nltk.download('wordnet',download_dir=nltk.data.path[0])
nltk.download('stopwords',download_dir=nltk.data.path[0])
nltk.download("words",download_dir=nltk.data.path[0])

def remove_stopwords_sentence(tokens):
    for word in tokens:
        if word in stopwords.words('english'):
            tokens.remove(word)
    
    return tokens

def is_gibberish(word):
    # Convert the word to lowercase for case-insensitive matching
    word = word.lower()

def text_to_tokens(text: str):
    tokenized_text = tokenize_document(text)
    print(tokenized_text)

    tokenized_text_no_gibberish = [word for word in tokenized_text if word in words.words() or word.lower() == "iphone"]

    text_nostop = remove_stopwords_sentence(tokenized_text_no_gibberish)
    print(text_nostop)

    if len(text_nostop) < 2:
        return []

    text_lemmatized = lemmatize_doc(text_nostop)
    print(text_lemmatized)

    tokens_bert = tokenizer(' '.join(text_lemmatized), return_tensors='pt')
    print(tokens_bert)

    return tokens_bert

def predict_review(text : str):
    tokens_bert = text_to_tokens(text)
    
    if len(tokens_bert) < 2:
        return 0, 1

    predictions = model(**tokens_bert)

    print(predictions.logits)

    # Apply softmax to output
    probabilities = F.softmax(predictions.logits, dim=-1)

    print(probabilities)

    # Apply softmax to output to get probabilities
    probabilities = torch.nn.functional.softmax(predictions.logits, dim=-1)

    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=-1)

    print(predicted_class)

    return int(predicted_class), float(max(probabilities[0]))

def get_sentiment(text):
    
    # initialize the VADER sentiment intensity analyzer
    sid = SentimentIntensityAnalyzer()

    lemmatized_text = lemmatize_doc(tokenize_document(text))

    sentiment_scores = sid.polarity_scores(' '.join(lemmatized_text))

    return sentiment_scores

# Set the page layout
st.set_page_config(layout="wide")

# Set the title
st.title("Smartphone Review and Sentiment Analysis")

# Text input
text = st.text_area("Write a text to analyze:")

# Predict button
if st.button("Analyze"):
    review = predict_review(text)

    # review
    if review[0] == 1:
        review_str = f"This is a review with probability {round(review[1]*100,1)}%."
    else:
        review_str = f"This is not a smartphone review with probability {round(review[1]*100,1)}%."
    
    # sentiment
    sentiment = get_sentiment(text)

    sentiment_str = f"The sentiment is {round(sentiment['neg']*100,1)}% negative, {round(sentiment['pos']*100,1)}% positive and {round(sentiment['neu']*100,1)}% neutral."

    st.write(review_str + "\n" + sentiment_str)