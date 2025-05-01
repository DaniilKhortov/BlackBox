from pymongo import MongoClient
from collections import Counter
import re

client = MongoClient("mongodb://localhost:27017/")
db = client["EnigmaticCodes"]
collection = db["PL"]

def pl_decrypt(pig_latin_word):
    if re.match(r"^[a-zA-Z]+$", pig_latin_word):
        if pig_latin_word.lower().endswith('yay'):
            # Випадок для слів, що починалися на голосну
            original_word = pig_latin_word[:-3] 
            return original_word
        elif pig_latin_word.lower().endswith('ay'):
            # Випадок для слів, що починалися на приголосну
            rest_of_word = pig_latin_word[:-2] 
            last_letter = rest_of_word[-1]
            original_word = last_letter + rest_of_word[:-1]
            return original_word
        else:
            return pig_latin_word 
    else:
        return pig_latin_word 

records = collection.find()
for record in records:
    decrypted = pl_decrypt(record["Encripted"])
    if decrypted != record["Original"]:
        print(f"Аномалія реверсивності: {record['Encripted']} -> {decrypted}, має бути {record['Original']}")