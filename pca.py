import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np
import re
import urllib.parse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopwords_set = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = urllib.parse.unquote(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stopwords_set]
    return ' '.join(tokens)

def read_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def extractLog(line):
    match = re.match(r"^(\S*)\s-\s-\s\[(.*?)\]\s\"(\S*)\s(\S*)\s([^\"]*)\"\s(\S*)\s(\S*)\s\"([^\"]*)\"\s\"([^\"]*)\"$", line)
    return match.groups() if match else [None] * 9

log_file_path = 'accesslog-all.log'
lines = read_log_file(log_file_path)
extracted_data = [extractLog(line) for line in lines]
urls = [data[3] for data in extracted_data]
preprocessed_urls = [preprocess(url) for url in urls]
tfidf_vectorizer = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_urls)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_df = tfidf_df.drop_duplicates()
tfidf_df = tfidf_df.loc[~(tfidf_df == 0).all(axis=1)]
pca = PCA(n_components=100)
pca_result = pca.fit_transform(tfidf_df)
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(100)])
pca_df.to_csv('accesslog-pca.csv', index=False)
print(pca_df.head())
explained_variance = pca.explained_variance_ratio_
print(f'explained variance ratio 10 components: {explained_variance[:10]}')
total_explained_variance = np.sum(explained_variance)
print(f'explained variance: {total_explained_variance}')
