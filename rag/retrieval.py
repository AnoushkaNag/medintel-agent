import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


INDEX_FILES = [
    "rag/vectorizer.pkl",
    "rag/embeddings.pkl",
    "rag/documents.pkl",
]


def ensure_index():

    if all(os.path.exists(path) for path in INDEX_FILES):
        return

    from rag.embeddings import create_embeddings

    create_embeddings()


def load_index():

    ensure_index()

    with open("rag/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("rag/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    with open("rag/documents.pkl", "rb") as f:
        docs = pickle.load(f)

    return vectorizer, embeddings, docs


def search(query, top_k=5):

    vectorizer, embeddings, docs = load_index()

    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, embeddings)[0]

    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in top_idx:

        results.append({
            "score": float(scores[i]),
            "document": docs[i]
        })

    return results


if __name__ == "__main__":

    results = search("hospitals with cardiology")

    for r in results:
        print(r["score"])
        print(r["document"])
        print("-" * 50)
