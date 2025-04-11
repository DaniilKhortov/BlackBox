import string

with open("text.txt", "r", encoding="utf-8") as file:
    content = file.read()
    words = content.split()

wordsUnique = {"seal", "walrus"}
for word in words:
    word = word.lower()
    word = word.replace('\u2060', '').replace(';', '').replace('?', '').replace('”', '').replace('—', '').replace('-', '').replace('“', '').replace('ä', 'a').replace('’', '').replace(':', '')
    word = word.strip(string.punctuation)  
    wordsUnique.add(word)

print(wordsUnique)
print(len(wordsUnique))
