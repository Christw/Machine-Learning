
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import json 
import re
import string

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
#pip3 install nltk
import nltk
from nltk.corpus import stopwords
#pip3 install spacy
import spacy
from spacy.lang.en.examples import sentences


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV



def main():
    ## Load the training data
    train = pd.DataFrame.from_records(json.load(open('train.json')))

    ## Functions
    # Connect first name and last name
    def connect_name(text):
        if text is not None:
            clean_list = []
            for i in list(text):
                clean_list.append(i.replace(', ','').replace(' ','').replace(',','').replace('-',''))
            return clean_list
    # Remove list
    def remove_list(text):
        return str(text).replace('[','').replace(']','')
    # Remove punctuation in the text
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase
    def lowercase_text(text):
        return text.lower()
    # Remove stopwords (English, French, Chinese) in the text
    def remove_unnecessary_words(text):
        all_stopwords = set(stopwords.words('english')+stopwords.words('french')+stopwords.words('chinese'))
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in all_stopwords]
        return ' '.join(filtered_words)

    ## Preprocess data using the above functions
    # Author_first_part
    train['author'] = train['author'].fillna(train['editor'])
    # General imputation with an empty text for each missing value
    train = train.fillna('')
    # Author_second_part
    train['author'] = train['author'].apply(connect_name) 
    train['author'] = train['author'].apply(remove_list)
    train['author'] = train['author'].apply(remove_punctuation)
    train['author'] = train['author'].apply(lowercase_text)
    # Publisher
    train['publisher'] = train['publisher'].apply(lowercase_text)
    # Title
    train['title'] = train['title'].str.replace(r'\[.*?\]', '', regex=True)
    train['title'] = train['title'].apply(remove_unnecessary_words)
    train['title'] = train['title'].apply(remove_punctuation)
    train['title'] = train['title'].apply(lowercase_text)
    # Abstract
    train['abstract'] = train['abstract'].str.replace(r'\[.*?\]', '', regex=True)
    train['abstract'] = train['abstract'].apply(remove_unnecessary_words)
    train['abstract'] = train['abstract'].apply(remove_punctuation)
    train['abstract'] = train['abstract'].apply(lowercase_text)
    # Editor
    train = train.drop('editor', axis=1)

    ## Split the data for training and testing
    X = train.drop(columns=["year"])
    y = train["year"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

    ## Transform text and categorical data into numeric values
    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm") # Download -> python3 -m spacy download en_core_web_sm
    # Tokenize the doc and lemmatize the tokens
    def my_tokenizer(doc): # Reference: https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af
        tokens = nlp(doc)
        return([token.lemma_ for token in tokens])
    # Define ColumnTransformer
    featurizer = ColumnTransformer(
        transformers=[
            ("title", HashingVectorizer(ngram_range=(1, 3),tokenizer = my_tokenizer,n_features =2**23), "title"),
            ("publisher", OneHotEncoder(handle_unknown='ignore'), ["publisher"]),
            ("ENTRYTYPE", OneHotEncoder(handle_unknown='ignore'), ["ENTRYTYPE"]),
            ("author", TfidfVectorizer(ngram_range=(1, 3),tokenizer = my_tokenizer), "author"),
            ("abstract", HashingVectorizer(ngram_range=(1, 3),tokenizer = my_tokenizer,n_features =2**23), "abstract")
        ])

    ## Create LinearSVC model and observate the performance before tuning
    lsvc = make_pipeline(featurizer, LinearSVC())
    lsvc.fit(X_train, y_train)
    err_base_lsvc = mean_absolute_error(y_test, lsvc.predict(X_test).astype(int))

    ## Hyperparameter tuning
    # Use GridSearch with cross validation to find the best setting and use the fine-tuned model to predict year
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 20.0],  # Regularization parameter
    }
    pipeline_lsvc = Pipeline([
        ('featurizer', featurizer),
        ('classifier', LinearSVC())
    ])
    grid_search_lsvc = GridSearchCV(pipeline_lsvc,param_grid,cv =5, scoring ='neg_mean_absolute_error' )
    grid_search_lsvc.fit(X_train,y_train)
    err_tuned_lsvc = mean_absolute_error(y_test, grid_search_lsvc.predict(X_test).astype(int))

    ## Final output
    # Load the testing data
    test = pd.DataFrame.from_records(json.load(open('test.json')))

    ## Preprocess the testing data like training data to make sure that they are in the same scale
    # Author_first_part
    test['author'] = test['author'].fillna(test['editor'])
    # General imputation with an empty text for each missing value
    test = test.fillna('')
    # Author_second_part
    test['author'] = test['author'].apply(connect_name) 
    test['author'] = test['author'].apply(remove_list)
    test['author'] = test['author'].apply(remove_punctuation)
    test['author'] = test['author'].apply(lowercase_text)
    # Publisher
    test['publisher'] = test['publisher'].apply(lowercase_text)
    # Title
    test['title'] = test['title'].str.replace(r'\[.*?\]', '', regex=True)
    test['title'] = test['title'].apply(remove_unnecessary_words)
    test['title'] = test['title'].apply(remove_punctuation)
    test['title'] = test['title'].apply(lowercase_text)
    # Abstract
    test['abstract'] = test['abstract'].str.replace(r'\[.*?\]', '', regex=True)
    test['abstract'] = test['abstract'].apply(remove_unnecessary_words)
    test['abstract'] = test['abstract'].apply(remove_punctuation)
    test['abstract'] = test['abstract'].apply(lowercase_text)
    # Editor
    test = test.drop('editor', axis=1)

    ## Predict year using the raw testing data
    y_pred_test = grid_search_lsvc.predict(test).astype(int)
    test['year'] = y_pred_test

    ## Output to json file
    final= test.drop(columns=['ENTRYTYPE', 'title','publisher','author','abstract'])
    final.to_json("predicted.json", orient='records', indent=2)

    return y_pred_test
    

main()