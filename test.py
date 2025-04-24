import string
def prepareData(sentence):
    words = sentence.split()
    output = []
    for word in words:
        word = word.lower()
        word = word.replace('\u2060', '').replace(';', '').replace('…', '').replace('ü', 'u').replace('ö', 'u').replace('!', '').replace(',', '').replace('?', '').replace('‘', '').replace('”', '').replace('—', '').replace('-', '').replace('“', '').replace('ä', 'a').replace('’', '').replace(':', '').replace(',', '')
        word = word.strip(string.punctuation)  
        output.append(word)
    return output        
inputSentence = "R fhvw gl ldm z xzhgov, mld rg rh ylcvh, gszg R szev g nlev"
print(prepareData(inputSentence))            
            