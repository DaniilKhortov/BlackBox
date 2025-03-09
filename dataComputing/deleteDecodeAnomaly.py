from pymongo import MongoClient
from collections import Counter


client = MongoClient("mongodb://localhost:27017/")
db = client["EnigmaticCodes"]
collection = db["Atbash"]

def atbash_decrypt(text):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    reverse_alphabet = alphabet[::-1]
    trans_table = str.maketrans(alphabet, reverse_alphabet)
    return text.translate(trans_table)

records = collection.find()
for record in records:
    decrypted = atbash_decrypt(record["Encripted"])
    if decrypted != record["Original"]:
        print(f"Аномалія реверсивності: {record['Encripted']} -> {decrypted}, має бути {record['Original']}")


