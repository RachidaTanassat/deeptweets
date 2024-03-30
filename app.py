import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model
loaded_model = joblib.load('tweet_classifier_model.joblib')

# Function for text preprocessing
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters and punctuation
    return text.strip()  # Remove leading/trailing whitespaces

def tokenize_text(text):
    return word_tokenize(text)  # Tokenize the text into words

stop_words = set(stopwords.words('english'))
def normalize_text(tokens):
    return [word.lower() for word in tokens if word.lower() not in stop_words]  # Convert to lowercase and remove stopwords

# Load the vocabulary for CountVectorizer
with open('count_vectorizer_vocabulary.joblib', 'rb') as f:
    vocabulary = joblib.load(f)

# Create a CountVectorizer instance for BoW and set its vocabulary
count_vectorizer = CountVectorizer(vocabulary=vocabulary)

# Title of the application
st.title("Tweet Classifier")

# Text input for user to enter tweet
tweet_input = st.text_input("Enter the tweet:")

# Button to classify the tweet
if st.button("Classify  :crystal_ball:"):
    # Clean, tokenize, and normalize the tweet
    cleaned_tweet = clean_text(tweet_input)
    tokenized_tweet = tokenize_text(cleaned_tweet)
    normalized_tweet = normalize_text(tokenized_tweet)

    # Convert the normalized tweet into BoW features
    bow_features_new = count_vectorizer.transform([' '.join(normalized_tweet)])

    # Predict label for the tweet
    predicted_label = loaded_model.predict(bow_features_new)
    st.write("Predicted Label:", predicted_label[0])
