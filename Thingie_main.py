# Using the Python API for Chat To RTX, this program will take a set of weblinks and then download & process a dataset that is exported to a set of Text files and launch ChatRTX with this new data set.
# The tool remains open in CMD where additional links can be added which will then be appended to the dataset.
# https://github.com/rpehkone/Chat-With-RTX-python-api
import rtx_api_3_5 as rtx_api
import os
import time
import requests
import json
import urllib.parse
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string

# Test RTX response
response = rtx_api.send_message("Say 'Hello'")
if "Hello" in response:
    print(response)
else:
    print("Something went wrong")


# Function to download and process a dataset from a list of weblinks
def download_and_process_dataset(weblinks):
    # Create an empty list to store the processed data
    processed_data = []

    # Loop through each weblink
    for weblink in weblinks:
        # Download the HTML content of the webpage
        try:
            response = requests.get(weblink)
            response.raise_for_status()  # Check for HTTP errors
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {weblink}: {e}")
            continue

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the text content from the webpage
        text = soup.get_text(separator=" ")

        # Preprocess the text data
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove non-alphanumeric characters
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Add the processed text to the list
        processed_data.append(tokens)

    # Return the processed data
    return processed_data


# Function to create a TF-IDF vectorizer
def create_tfidf_vectorizer(processed_data):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the processed data
    vectorizer.fit(processed_data)

    # Return the vectorizer
    return vectorizer


# Function to calculate the similarity between two sentences
def calculate_similarity(sentence1, sentence2, vectorizer):
    # Convert the sentences to vectors
    vector1 = vectorizer.transform([sentence1])
    vector2 = vectorizer.transform([sentence2])

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)[0][0]

    # Return the similarity score
    return similarity


# Function to generate a response based on the user's input
def generate_response(user_input, processed_data, vectorizer):
    # Calculate the similarity between the user's input and each sentence in the dataset
    similarities = []
    for sentence in processed_data:
        similarity = calculate_similarity(user_input, sentence, vectorizer)
        similarities.append(similarity)

    # Find the sentence with the highest similarity score
    most_similar_index = similarities.index(max(similarities))

    # Generate a response based on the most similar sentence
    response = processed_data[most_similar_index]

    # Return the response
    return response


# Main function
def main():
    # Get the list of weblinks from the user
    weblinks = []
    while True:
        weblink = input("Enter a weblink (or 'q' to quit): ")
        if weblink == "       q":
            break
        weblinks.append(weblink)

    # Download and process the dataset
    processed_data = download_and_process_dataset(weblinks)

    # Create a TF-IDF vectorizer
    vectorizer = create_tfidf_vectorizer(processed_data)

    # Start the ChatRTX session
    while True:
        # Get the user's input
        user_input = input("You: ")

        # Generate a response
        response = generate_response(user_input, processed_data, vectorizer)

        # Print the response
        print("ChatRTX:", response)


# Run the main function
if __name__ == "__main__":
    main()
