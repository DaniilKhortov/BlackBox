from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["EnigmaticCodes"]
collection = db["Atbash"]


records = collection.find()

for record in records:
    original = record["Original"]
    encrypted = record["Encripted"]
    
    if len(original) != len(encrypted):
        print(f"Аномалія довжини: {original} -> {encrypted}")
