import string, os


with open(os.path.abspath("text.txt"), "r", encoding="utf-8") as file:
    content = file.read()
    words = content.split()

wordsUnique = {"seal", "walrus"}
for word in words:
    word = word.lower()
    word = word.replace('\u2060', '').replace(';', '').replace('…', '').replace('ü', 'u').replace('ö', 'u').replace('!', '').replace(',', '').replace('?', '').replace('‘', '').replace('”', '').replace('—', '').replace('-', '').replace('“', '').replace('ä', 'a').replace('’', '').replace(':', '')
    word = word.strip(string.punctuation)  
    wordsUnique.add(word)

print(wordsUnique)
print(len(wordsUnique))
