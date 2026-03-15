import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def build_documents(df):
    docs = []

    for _, row in df.iterrows():

        text = f"""
        Facility: {row['facility']}
        Region: {row['region']}
        Specialties: {row['specialties']}
        Procedures: {row['procedures']}
        Equipment: {row['equipment']}
        Capabilities: {row['capabilities']}
        """

        docs.append(text)

    return docs


def create_embeddings():

    df = pd.read_csv("data/structured_capabilities_geo.csv")

    docs = build_documents(df)

    vectorizer = TfidfVectorizer(stop_words="english")

    embeddings = vectorizer.fit_transform(docs)

    with open("rag/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("rag/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    with open("rag/documents.pkl", "wb") as f:
        pickle.dump(docs, f)

    print("Embeddings created successfully")


if __name__ == "__main__":
    create_embeddings()