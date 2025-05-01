from Classificator import decrypt_sentence_interface, initialize_system

initialize_system()
print("\n" + "="*20 + " ТЕСТУВАННЯ ІНТЕРФЕЙСУ " + "="*20)
# ... (ваш тестовий блок залишається тим самим) ...
test_sentences = [
    "Z nlxp gvhg gl xsvxp wvxlwrmt.",             # Atbash
    "B npdl uftu up difdl efdpejoh.",             # Caesar (shift 1)
    "E qsgo xiwx xs gligo higshmrk.",             # Caesar (shift 4)
    "Igpay Atinlay isyay unfay!",                 # Pig Latin
    "esttay ordway",                              # Pig Latin (невідомі слова)
    "This is a plain text sentence.",             # Не шифрований
    "Це речення кирилицею!",                     # Недозволені символи
    "",                                           # Порожній рядок
    "   ",                                        # Тільки пробіли
    "123 4567",                                   # Тільки цифри і пробіли
    "word 123 test!",                             # Змішаний валідний
    "invalid-char#"                               # Недозволені символи
]
for sentence in test_sentences:
    cipher_type, result, msgs = decrypt_sentence_interface(sentence)
    print(f"\n--- Вхід: '{sentence}' ---")
    print(f"Тип шифру: {cipher_type}")
    print(f"Результат: '{result}'")
    print("Повідомлення:")
    # Фільтруємо повідомлення для кращої читабельності (прибираємо DEBUG, ІНФО)
    # або виводимо всі
    for msg in msgs:
        print(f"  - {msg}")
    print("-" * 60)

"""initialize_system()
print("\n" + "="*20 + " ТЕСТУВАННЯ ІНТЕРФЕЙСУ " + "="*20)
test_sentences = [
    "Z nlxp gvhg gl xsvxp wvxlwrmt.", # Atbash
    "B npdl uftu up difdl efdpejoh.", # Caesar (shift 1)
    "E qsgo xiwx xs gligo higshmrk.", # Caesar (shift 4)
    "Igpay Atinlay isyay unfay!",     # Pig Latin
    "Ayay ockmay esttay otay heckcay ecodingday", # Pig Latin 
    "Ой у лузі червона калина." # Не шифрований
]
for sentence in test_sentences:
    cipher_type, result = decrypt_sentence_interface(sentence)
    print(f"Результат інтерфейсу: Тип={cipher_type}, Розшифровано='{result}'")
    print("-"*60)"""

 #A mock test to check decoding.
#"Z nlxp gvhg gl xsvxp wvxlwrmt." - atbash
#"Ayay ockmay esttay otay heckcay ecodingday" - pl

#B npdl uftu up difdl efdpejoh. - caesar1
#C oqem vguv vq ejgem fgeqfkpi. - caesar2
#D prfn whvw wr fkhfn ghfrglqj. - caesar3
#E qsgo xiwx xs gligo higshmrk. - caesar4    