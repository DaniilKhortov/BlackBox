from pymongo import MongoClient

# Підключення до MongoDB
client = MongoClient("mongodb://localhost:27017/")  
db = client["EnigmaticCodes"]
# collection = db["Atbash"]
collection = db["Caesar"]

pipeline = [
    {"$group": {
        "_id": "$Original", 
        "docs": {"$push": "$_id"},
        "count": {"$sum": 1}
    }},
    {"$match": {"count": {"$gt": 1}}}  
]

duplicates = list(collection.aggregate(pipeline))


for entry in duplicates:
    doc_ids = entry["docs"][1:]  
    collection.delete_many({"_id": {"$in": doc_ids}})

print("Дублікати успішно видалено!")
