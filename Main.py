# ---------------------------
# IMPORTS
# ---------------------------
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv('CourseraDataset-Clean.csv')
df = df.fillna('')

df['Text'] = df['Course Title'] + ' ' + df['What you will learn'] + ' ' + df['Skill gain']

# ---------------------------
# NLTK SETUP
# ---------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# PREPROCESS FUNCTION
# ---------------------------
def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [re.sub(r'[^a-zA-Z]', '', t) for t in tokens if re.sub(r'[^a-zA-Z]', '', t)]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['Text'].apply(preprocess)

# ---------------------------
# QUERY EXPANSION
# ---------------------------
def expand_query(text):
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    tokens = [re.sub(r'[^a-zA-Z]', '', t) for t in tokens if re.sub(r'[^a-zA-Z]', '', t)]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]

    expanded = set(tokens)

    for t in tokens:
        for syn in wordnet.synsets(t):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_',' '))

    return ' '.join(expanded)

# ---------------------------
# TF-IDF + KNN (GLOBAL MODEL)
# ---------------------------
vectorizer = TfidfVectorizer(max_features=7000)
x = vectorizer.fit_transform(df['clean_text'])

knn = NearestNeighbors(n_neighbors=20, metric='cosine')
knn.fit(x)

# ---------------------------
# RECOMMEND FUNCTION
# ---------------------------
def recommend(query):

    expanded_query = expand_query(query)
    query_vec = vectorizer.transform([expanded_query])

    # KNN
    distance, indices = knn.kneighbors(query_vec)
    knn_indices = indices[0]

    return df.iloc[knn_indices]


# ---------------------------
# FEEDBACK FUNCTION
# ---------------------------
def feedback(query, knn_indices):

    expanded_query = expand_query(query)
    query_vec = vectorizer.transform([expanded_query])

    # Rocchio
    alpha, beta, gamma = 1, 0.75, 0.15

    relevant_idx = knn_indices[:10]
    non_relevant_idx = knn_indices[10:]

    # Dr = x[relevant_idx].mean(axis=0)
    # Dnr = x[non_relevant_idx].mean(axis=0)
    if len(relevant_idx) > 0:
        Dr = x[relevant_idx].mean(axis=0)
    else:
        Dr = query_vec  # fallback
    # Non-relevant mean
    if len(non_relevant_idx) > 0:
        Dnr = x[non_relevant_idx].mean(axis=0)
    else:
        Dnr = csr_matrix(query_vec.shape)  # zero vector

    Dr = csr_matrix(Dr)
    Dnr = csr_matrix(Dnr)

    new_query_vec = query_vec.multiply(alpha)
    new_query_vec = new_query_vec + Dr.multiply(beta)
    new_query_vec = new_query_vec - Dnr.multiply(gamma)

    # Final similarity
    final_scores = cosine_similarity(new_query_vec, x)[0]

    # Ranking factors
    df['rating_norm'] = df['Rating']/5.0
    df['review_norm'] = np.log1p(df['Number of Review']) / np.log1p(df['Number of Review']).max()

    final_score = (
        0.6 * final_scores +
        0.25 * df['rating_norm'] +
        0.15 * df['review_norm']
    )

    final_indices = np.argsort(final_score)[-10:][::-1]

    return df.iloc[final_indices]