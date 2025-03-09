from pymongo import MongoClient
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np

client = MongoClient("mongodb://localhost:27017/")
db = client["EnigmaticCodes"]
collection = db["Atbash"]



records = list(collection.find())

vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2))
X = vectorizer.fit_transform([r["Encripted"] for r in records])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

k = 0
for i, record in enumerate(records):
    print(f"{record['Original']} -> {record['Encripted']} (Кластер {labels[i]})")
    k+=1

