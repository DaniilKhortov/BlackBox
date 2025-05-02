from toLowerCase import wordsUnique
import csv

lookupTable = {'A' : 'Z', 'B' : 'Y', 'C' : 'X', 'D' : 'W', 'E' : 'V',
        'F' : 'U', 'G' : 'T', 'H' : 'S', 'I' : 'R', 'J' : 'Q',
        'K' : 'P', 'L' : 'O', 'M' : 'N', 'N' : 'M', 'O' : 'L',
        'P' : 'K', 'Q' : 'J', 'R' : 'I', 'S' : 'H', 'T' : 'G',
        'U' : 'F', 'V' : 'E', 'W' : 'D', 'X' : 'C', 'Y' : 'B', 'Z' : 'A',
        'a' : 'z', 'b' : 'y', 'c' : 'x', 'd' : 'w', 'e' : 'v',
        'f' : 'u', 'g' : 't', 'h' : 's', 'i' : 'r', 'j' : 'q',
        'k' : 'p', 'l' : 'o', 'm' : 'n', 'n' : 'm', 'o' : 'l',
        'p' : 'k', 'q' : 'j', 'r' : 'i', 's' : 'h', 't' : 'g',
        'u' : 'f', 'v' : 'e', 'w' : 'd', 'x' : 'c', 'y' : 'b', 'z' : 'a'}

def atbash(message):
    cipher = ''
    for letter in message:

        if(letter != ' '):

            cipher += lookupTable[letter]
        else:

            cipher += ' '
 
    return cipher

data = []

for word in wordsUnique:
    print ("Text  : " + word)
    print ("Cipher: " + atbash(word))
    
    mydict = { "Original": word, "Encripted": atbash(word) }
    data.append(mydict)
    
with open('dataA.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["Original","Encripted"])
    writer.writeheader()  
    writer.writerows(data)   