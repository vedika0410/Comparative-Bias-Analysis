from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


def tokenize(text_series):
    return [text.lower().split() for text in text_series]


def sentence_vector(model, tokens, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def train_word2vec_model(df):
    texts = df["hard_text"]
    y = df["income_binary"]

    tokenized_text = tokenize(texts)

    # Train Word2Vec
    w2v_model = Word2Vec(
        sentences=tokenized_text,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=10
    )

    # Convert sentences to vectors
    X = np.array([
        sentence_vector(w2v_model, tokens, 100)
        for tokens in tokenized_text
    ])

    # Use index-aware split
    indices = df.index

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return clf, test_idx, y_test, y_pred