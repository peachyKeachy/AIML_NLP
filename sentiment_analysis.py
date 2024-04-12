import pandas as pd
# nltk is a NLP library 

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# preprocess_text function
def preprocess_text(text):

    # tokenize text (creating vocabulary) and put into lower case (.lower)
    tokens = word_tokenize(text.lower())
    
    # remove stop words (words we dont want) with list comprehension
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # join tokens back into a string
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

# download functions from nltk
nltk.download('all')

# importing Amazon Review Dataset
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

print('AMAZON REVIEW DATASET: ')
print(df.head())

# preprocess review text
df['reviewText'] = df['reviewText'].apply(preprocess_text)

print(df)

# initialize NLTK sentiment analyzer 
analyzer = SentimentIntensityAnalyzer()

# get_sentiment function
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment 

# apply get_sentiment function
df['sentiment'] = df['reviewText'].apply(get_sentiment)

print(df)
