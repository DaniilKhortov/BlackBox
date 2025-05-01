import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import Levenshtein # Потрібно встановити: pip install python-Levenshtein
import statistics # Для обчислення середнього

# (Код функцій залишається таким же, як у попередній відповіді)

def load_dictionary(file_path="D:/Mysor2/3kurs/Coll_proc_data/Project/dataPL_r.csv"):
    #EnigmaticCodes.PL
    #EnigmaticCodes.Atbash


    #dataA_r
    #dataPL_r
    """
    Завантажує словник шифрування з CSV файлу, де дані записані у форматі:
    зашифроване_слово,дешифроване_слово (приклад: mut,nfg).
    """
    dictionary = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Пропускаємо заголовок, якщо він є
            # Визначаємо індекси стовпців (адаптовано до можливих назв)
            try:
                header_lower = [h.lower().strip() for h in header]
                # Спробуємо знайти очікувані стовпці
                try:
                     decrypted_col_index = header_lower.index("original") # або "decrypted"
                except ValueError:
                     decrypted_col_index = 0 # Припускаємо перший стовпець

                try:
                     encrypted_col_index = header_lower.index("encripted") # або "encrypted"
                except ValueError:
                    # Припускаємо останній стовпець, якщо не знайдено "encripted"
                    encrypted_col_index = len(header_lower) - 1
                    if len(header_lower) == 3 and 'slide' in header_lower: # Спеціальний випадок для Цезаря
                        encrypted_col_index = 2

                print(f"Зчитано заголовок: {header}")
                print(f"Індекс стовпця Original/Decrypted: {decrypted_col_index}")
                print(f"Індекс стовпця Encripted/Encrypted: {encrypted_col_index}")

            except Exception as e_header:
                 print(f"Помилка визначення стовпців у заголовку: {e_header}. Перевірте заголовок CSV.")
                 return None


            for row_num, row in enumerate(csv_reader, start=2):
                if len(row) > max(decrypted_col_index, encrypted_col_index): # Перевірка наявності потрібних стовпців
                    try:
                        decrypted_word = row[decrypted_col_index].strip()
                        encrypted_word = row[encrypted_col_index].strip()

                        if not decrypted_word or not encrypted_word:
                             print(f"Попередження: Пропущено рядок {row_num} через порожнє слово: {row}")
                             continue

                        dictionary[encrypted_word] = decrypted_word # Записуємо у словник: зашифроване -> дешифроване
                    except IndexError:
                        print(f"Попередження: Помилка індексу при обробці рядка {row_num}: {row}")
                else:
                    print(f"Попередження: Пропущено рядок {row_num} через недостатню кількість стовпців: {row}")
    except FileNotFoundError:
        print(f"Помилка: Файл словника '{file_path}' не знайдено.")
        return None
    except Exception as e:
        print(f"Помилка при читанні CSV файлу '{file_path}': {e}") # Обробка інших можливих помилок при читанні CSV
        return None

    if not dictionary:
        print(f"Попередження: Словник порожній після завантаження файлу {file_path}.")
    return dictionary

def preprocess_data(dictionary, input_characters=None, target_characters=None):
    """Підготовка даних для RNN: створення словників символів, векторизація, padding."""
    encrypted_words = list(dictionary.keys())
    decrypted_words = list(dictionary.values())

    # Створюємо словники символів IF NOT PROVIDED
    if input_characters is None:
        input_characters = set()
        for word in encrypted_words:
            for char in word:
                input_characters.add(char)
        input_characters = sorted(list(input_characters))
        # print(f"preprocess_data (initial) - Calculated input_characters: {input_characters}")

    if target_characters is None:
        target_characters = set()
        for word in decrypted_words:
            for char in word:
                target_characters.add(char)
        target_characters = sorted(list(target_characters))
        # print(f"preprocess_data (initial) - Calculated target_characters: {target_characters}")

    # Додаємо спеціальні символи, якщо їх немає
    input_chars_list = list(input_characters) # Працюємо зі списком для append/insert
    target_chars_list = list(target_characters)

    if ' ' not in input_chars_list:
        input_chars_list.append(' ')
        input_chars_list.sort()
    if ' ' not in target_chars_list:
        target_chars_list.append(' ')
        target_chars_list.sort()
    if '\t' not in target_chars_list: # Start token
        target_chars_list.insert(0, '\t') # Вставляємо на початок
    if '\n' not in target_chars_list: # End token
        target_chars_list.append('\n')

    # Оновлені списки символів
    input_characters_final = input_chars_list
    target_characters_final = target_chars_list


    num_encoder_tokens = len(input_characters_final)
    num_decoder_tokens = len(target_characters_final)
    # Обчислюємо max_len безпечно, перевіряючи на порожній список
    max_encoder_seq_length = max([len(word) for word in encrypted_words]) if encrypted_words else 0
    max_decoder_seq_length = max([len(word) for word in decrypted_words]) + 1 if decrypted_words else 1 # +1 для \t та \n

    print("Кількість унікальних вхідних символів:", num_encoder_tokens)
    print("Кількість унікальних вихідних символів:", num_decoder_tokens)
    print("Макс. довжина вхідної послідовності:", max_encoder_seq_length)
    print("Макс. довжина вихідної послідовності:", max_decoder_seq_length)
    # print("Вхідні символи:", input_characters_final)
    # print("Вихідні символи:", target_characters_final)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters_final)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters_final)])

    # Перевіряємо, чи є дані для обробки
    if not encrypted_words:
         print("Попередження: Немає даних для обробки в preprocess_data.")
         # Повертаємо порожні структури та нульові значення
         return (np.zeros((0, 0, 0)), np.zeros((0, 0, 0)), np.zeros((0, 0, 0)),
                 0, 0, {}, {}, [], [], 0, 0)


    encoder_input_data = np.zeros(
        (len(encrypted_words), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(decrypted_words), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(decrypted_words), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    space_token_index_enc = input_token_index.get(' ', -1) # Отримуємо індекс пробілу безпечно
    space_token_index_dec = target_token_index.get(' ', -1)

    if space_token_index_enc == -1 or space_token_index_dec == -1:
        print("Помилка: Не вдалося знайти індекс для символу пробілу (' '). Перевірте словники символів.")
        # Можна повернути помилку або порожні дані
        return (np.zeros((0, 0, 0)), np.zeros((0, 0, 0)), np.zeros((0, 0, 0)),
                 0, 0, {}, {}, [], [], 0, 0)


    for i, (input_text, target_text) in enumerate(dictionary.items()):
        for t, char in enumerate(input_text):
            if t < max_encoder_seq_length and char in input_token_index:
                encoder_input_data[i, t, input_token_index[char]] = 1.0
        # Padding для решти послідовності
        if len(input_text) < max_encoder_seq_length:
             encoder_input_data[i, len(input_text):, space_token_index_enc] = 1.0

        target_text_with_tokens = '\t' + target_text + '\n'
        for t, char in enumerate(target_text_with_tokens):
            if t < max_decoder_seq_length and char in target_token_index:
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        # Padding для решти послідовності
        if len(target_text_with_tokens) < max_decoder_seq_length:
            decoder_input_data[i, len(target_text_with_tokens):, space_token_index_dec] = 1.0
        if len(target_text_with_tokens) -1 < max_decoder_seq_length:
            decoder_target_data[i, len(target_text_with_tokens)-1:, space_token_index_dec] = 1.0


    return encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index, input_characters_final, target_characters_final, max_encoder_seq_length, max_decoder_seq_length

def build_seq2seq_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    """Створює модель Seq2Seq RNN з ОДНИМ шаром LSTM."""
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens), name='encoder_input')
    encoder_lstm = keras.layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_inputs) # Ігноруємо вихід послідовності енкодера
    encoder_states = [state_h, state_c]

    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens), name='decoder_input')
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax", name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=64, epochs=100, validation_split=0.2):
    """Навчає модель Seq2Seq."""
    print(f"train_model - encoder_input_data.shape: {encoder_input_data.shape}, decoder_input_data.shape: {decoder_input_data.shape}, decoder_target_data.shape: {decoder_target_data.shape}")
    # Перевірка на нульовий розмір перед навчанням
    if encoder_input_data.shape[0] == 0 or decoder_input_data.shape[0] == 0 or decoder_target_data.shape[0] == 0:
        print("Помилка: Навчальні дані порожні. Навчання неможливе.")
        return model, None # Повертаємо модель без навчання та порожню історію

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
    )
    return model, history

def create_inference_models(model, latent_dim, num_decoder_tokens): # Оновлено параметри
    """Створює окремі моделі для висновування."""
    encoder_inputs = model.input[0]  # encoder_inputs
    _, state_h_enc, state_c_enc = model.get_layer('encoder_lstm').output # encoder_lstm states
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs_inf = keras.Input(shape=(1, num_decoder_tokens)) # Вхід для одного символу
    decoder_state_input_h = keras.Input(shape=(latent_dim,))
    decoder_state_input_c = keras.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs_inf, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs_inf, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.get_layer('decoder_dense')
    decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
    decoder_model = keras.Model(
        [decoder_inputs_inf] + decoder_states_inputs, [decoder_outputs_inf] + decoder_states
    )

    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens):
    """Декодує введену послідовність."""
    states_value = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Перевірка наявності '\t' у словнику
    if '\t' not in target_token_index:
        print("Помилка: Символ початку послідовності '\\t' не знайдено у target_token_index.")
        return "" # Повертаємо порожній рядок у випадку помилки
    target_seq[0, 0, target_token_index['\t']] = 1.0

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index.get(sampled_token_index, '') # Безпечне отримання

        # Умова зупинки
        if sampled_char == '\n' or len(decoded_sentence) >= max_decoder_seq_length - 1:
            stop_condition = True
            continue # Не додаємо символ кінця рядка до результату

        decoded_sentence += sampled_char

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return decoded_sentence

def calculate_word_similarity(decrypted_word, dictionary_words):
    """Визначає схожість дешифрованого слова."""
    min_distance = float('inf')
    closest_word = None
    if not dictionary_words:
        return min_distance, closest_word
    for dict_word in dictionary_words:
        if not isinstance(decrypted_word, str) or not isinstance(dict_word, str):
             # print(f"Попередження: Некоректні типи для Levenshtein: '{decrypted_word}' ({type(decrypted_word)}), '{dict_word}' ({type(dict_word)})")
             continue
        distance = Levenshtein.distance(decrypted_word, dict_word)
        if distance < min_distance:
            min_distance = distance
            closest_word = dict_word
    return min_distance, closest_word

# --- НОВА ФУНКЦІЯ ОЦІНКИ ---
def evaluate_model_performance(encoder_input_data_test, dictionary_test,
                              encoder_model, decoder_model, input_token_index, target_token_index, # input_token_index все ще потрібен для encode, але не для decode
                              reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens,
                              english_dictionary_words):
    """
    Оцінює продуктивність моделі дешифрування на тестових даних.
    """
    correct_word_count = 0
    total_words = len(encoder_input_data_test)
    if total_words == 0:
        print("Попередження: Тестовий набір даних порожній. Оцінка неможлива.")
        return 0, float('inf'), 0

    levenshtein_distances = []
    total_expected_chars = 0
    total_correct_chars = 0

    input_words_list_test = list(dictionary_test.keys())

    print(f"\n--- Початок оцінки на {total_words} тестових словах ---")

    for i in range(total_words):
        input_seq = encoder_input_data_test[i: i + 1]
        input_word = input_words_list_test[i]
        expected_decoding = dictionary_test[input_word]

        # --- ВАЖЛИВО: Переконайтесь, що decode_sequence існує і приймає ці аргументи ---
        try:
            decoded_sentence = decode_sequence(
                input_seq, encoder_model, decoder_model,
                # input_token_index, # **ВИДАЛЕНО input_token_index**
                target_token_index,
                reverse_target_char_index,
                max_decoder_seq_length, num_decoder_tokens
            )
        except NameError:
             print("Помилка: Функція decode_sequence не визначена або недоступна.")
             continue # Пропускаємо поточну ітерацію, якщо функція не знайдена
        except Exception as e_decode: # Ловимо інші можливі помилки декодування
            print(f"Помилка під час декодування слова '{input_word}': {e_decode}")
            continue

        decoded_word = decoded_sentence.strip().lower()
        expected_decoding_lower = expected_decoding.lower()

        # 1. Word Accuracy
        if decoded_word == expected_decoding_lower:
            correct_word_count += 1

        # 2. Levenshtein Distance
        if isinstance(decoded_word, str) and isinstance(expected_decoding_lower, str):
             distance = Levenshtein.distance(decoded_word, expected_decoding_lower)
             levenshtein_distances.append(distance)
        else:
             print(f"Попередження: Некоректні типи для Levenshtein при оцінці: '{decoded_word}', '{expected_decoding_lower}'")


        # 3. Character Accuracy
        total_expected_chars += len(expected_decoding_lower)
        for j in range(min(len(decoded_word), len(expected_decoding_lower))):
            if decoded_word[j] == expected_decoding_lower[j]:
                total_correct_chars += 1

        # Виводимо прогрес
        if (i + 1) % 100 == 0 or (i + 1) == total_words:
            print(f"Оброблено {i + 1}/{total_words} слів...")

    word_accuracy = (correct_word_count / total_words) * 100 if total_words > 0 else 0
    avg_levenshtein_distance = statistics.mean(levenshtein_distances) if levenshtein_distances else float('inf')
    char_accuracy = (total_correct_chars / total_expected_chars) * 100 if total_expected_chars > 0 else 0

    print("--- Оцінку завершено ---")
    return word_accuracy, avg_levenshtein_distance, char_accuracy


# --- Головний блок ---
if __name__ == "__main__":
    # 1. Завантаження словника
    dictionary_full = load_dictionary()
    if dictionary_full is None or not dictionary_full: # Додаткова перевірка на порожній словник
        print("Помилка: Словник порожній або не вдалося завантажити. Завершення роботи.")
        exit()

    # 2. Розділення даних на навчальні та тестові
    all_encrypted_words = list(dictionary_full.keys())
    if len(all_encrypted_words) < 2: # Перевірка на достатню кількість даних для розділення
         print("Помилка: Недостатньо даних у словнику для розділення на навчальний та тестовий набори.")
         exit()

    train_encrypted, test_encrypted = train_test_split(all_encrypted_words, test_size=0.05, random_state=42)

    dictionary_train = {k: dictionary_full[k] for k in train_encrypted}
    dictionary_test = {k: dictionary_full[k] for k in test_encrypted}

    # 3. Підготовка навчальних даних
    print("\n--- Підготовка навчальних даних ---")
    (encoder_input_data_train, decoder_input_data_train, decoder_target_data_train,
     num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index,
     input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length) = preprocess_data(dictionary_train)

    if encoder_input_data_train.size == 0:
        print("Помилка: Не вдалося підготувати навчальні дані.")
        exit()

    # Створення зворотного словника для вихідних символів
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    # 4. Підготовка тестових даних
    print("\n--- Підготовка тестових даних ---")
    (encoder_input_data_test, _, _, _, _, _, _, _, _, _, _) = preprocess_data(
        dictionary_test,
        input_characters=input_characters, # Передаємо словники з навчальних даних
        target_characters=target_characters
    )
    if encoder_input_data_test.size == 0:
         print("Помилка: Не вдалося підготувати тестові дані.")
         # Можна продовжити без оцінки, якщо хочете
         # exit()


     # 5. Побудова моделі Seq2Seq
    latent_dim = 256
    model = build_seq2seq_model(num_encoder_tokens, num_decoder_tokens, latent_dim)
    print("\n--- Структура моделі ---")
    model.summary()
    try:
        keras.utils.plot_model(model, to_file='seq2seq_model.png', show_shapes=True)
        print("Структуру моделі збережено у seq2seq_model.png")
    except ImportError as e:
        print(f"Попередження: Не вдалося візуалізувати модель. Помилка: {e}.")

    # 6. Навчання моделі
    print("\n--- Початок навчання моделі ---")
    epochs = 100
    batch_size = 128
    model, training_history = train_model(model, encoder_input_data_train, decoder_input_data_train, decoder_target_data_train, epochs=epochs, batch_size=batch_size)

    # Перевірка, чи навчання відбулося
    if training_history is None:
         print("Помилка: Навчання моделі не відбулося.")
         exit()
    print("--- Навчання моделі завершено ---")

    # 7. Створення моделей для висновування
    print("\n--- Створення моделей для висновування ---")
    try:
        encoder_model, decoder_model = create_inference_models(
            model, latent_dim, num_decoder_tokens
        )
        print("Моделі для висновування створено.")
    except Exception as e_inf:
        print(f"Помилка створення моделей для висновування: {e_inf}")
        exit()


    # 8. Побудова графіків навчання
    print("\n--- Побудова графіків навчання ---")
    if training_history and 'accuracy' in training_history.history and 'val_accuracy' in training_history.history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(training_history.history['accuracy'])
        plt.plot(training_history.history['val_accuracy'])
        plt.title('Точність моделі')
        plt.ylabel('Точність')
        plt.xlabel('Епоха')
        plt.legend(['Навчання', 'Валідація'], loc='upper left')
    else:
        print("Попередження: Дані про точність відсутні в історії навчання.")

    if training_history and 'loss' in training_history.history and 'val_loss' in training_history.history:
        plt.subplot(1, 2, 2)
        plt.plot(training_history.history['loss'])
        plt.plot(training_history.history['val_loss'])
        plt.title('Втрати моделі')
        plt.ylabel('Втрати')
        plt.xlabel('Епоха')
        plt.legend(['Навчання', 'Валідація'], loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
         print("Попередження: Дані про втрати відсутні в історії навчання.")


    # 9. Завантаження списку "справжніх" англійських слів (словник)
    english_dictionary_file = "D:/Mysor2/3kurs/Coll_proc_data/Project/words_alphaPL_r.txt" # Змінено на загальний словник
    english_dictionary_words = []
    try:
        with open(english_dictionary_file, 'r', encoding='utf-8') as word_file:
            english_dictionary_words = [word.strip().lower() for word in word_file]
        print(f"\nЗавантажено {len(english_dictionary_words)} слів з словника {english_dictionary_file}")
    except FileNotFoundError:
        print(f"\nПомилка: Файл словника англійських слів '{english_dictionary_file}' не знайдено.")

    # 10. Оцінка продуктивності на тестових даних
    if encoder_input_data_test.size > 0: # Перевірка чи є тестові дані
        word_acc, avg_lev_dist, char_acc = evaluate_model_performance(
            encoder_input_data_test, dictionary_test,
            encoder_model, decoder_model, input_token_index, target_token_index,
            reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens,
            english_dictionary_words
        )

        print("\n--- Оцінка продуктивності моделі на тестових даних ---")
        print(f"Точність на рівні слів: {word_acc:.2f}%")
        print(f"Середня відстань Левенштейна: {avg_lev_dist:.2f}")
        print(f"Точність на рівні символів: {char_acc:.2f}%")
    else:
        print("\nПопередження: Тестові дані порожні, оцінка продуктивності не проводиться.")


    # 11. Тестування дешифрування (кілька прикладів з тестового набору)
    print("\n--- Приклади дешифрування (тестовий набір) ---")
    num_display_words = min(10, len(encoder_input_data_test))
    if num_display_words > 0:
        test_indices = np.random.choice(len(encoder_input_data_test), num_display_words, replace=False)

        for i, seq_index in enumerate(test_indices):
            input_seq = encoder_input_data_test[seq_index: seq_index + 1]
            # Отримуємо відповідне зашифроване слово з тестового набору
            input_word = list(dictionary_test.keys())[seq_index] # Використовуємо індекс тестового набору
            expected_decoding = dictionary_test[input_word]

            decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens)
            decoded_word_clean = decoded_sentence.strip().lower()
            similarity_distance, closest_dict_word = calculate_word_similarity(decoded_word_clean, english_dictionary_words)

            print('-')
            print(f"Зашифроване слово: {input_word}")
            print(f"Дешифроване слово (RNN): {decoded_word_clean}")
            print(f"Очікуване дешифрування: {expected_decoding}")
            if english_dictionary_words:
                print(f"Відстань Левенштейна до найближчого слова: {similarity_distance}, Найближче слово: {closest_dict_word}")
    else:
        print("Немає тестових даних для відображення прикладів.")


    # 12. Дешифрування невідомих слів (приклад)
    unknown_encrypted_words = ["ergzorp", "wzmrro", "rihosre"] # vitalik, daniil, irslhiv
    print("\n--- Дешифрування невідомих слів ---")
    for encrypted_word in unknown_encrypted_words:
        test_input_data = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        word_len = 0
        valid_word = True
        for t, char in enumerate(encrypted_word):
            if t < max_encoder_seq_length:
                if char in input_token_index:
                    test_input_data[0, t, input_token_index[char]] = 1.0
                    word_len += 1
                else:
                    print(f"Попередження: Невідомий символ '{char}' у слові '{encrypted_word}'. Дешифрування може бути неточним.")
                    # Можна або пропустити слово, або продовжити без цього символу
                    valid_word = False # Позначимо, що слово має невідомі символи
                    # break # або зупинити обробку цього слова
            else:
                 print(f"Попередження: Слово '{encrypted_word}' довше за max_encoder_seq_length ({max_encoder_seq_length}). Обрізано.")
                 break # Обрізаємо слово

        # Padding
        if word_len < max_encoder_seq_length and ' ' in input_token_index:
             test_input_data[0, word_len:, input_token_index[' ']] = 1.0

        if not valid_word: # Якщо були невідомі символи, можемо пропустити дешифрування
             print(f"Пропускаємо дешифрування слова '{encrypted_word}' через невідомі символи.")
             continue

        decoded_sentence = decode_sequence(test_input_data, encoder_model, decoder_model, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens)
        decoded_word_clean = decoded_sentence.strip().lower()
        similarity_distance, closest_dict_word = calculate_word_similarity(decoded_word_clean, english_dictionary_words)

        print(f"Зашифроване слово: {encrypted_word}, Дешифроване слово (RNN): {decoded_word_clean}", end="")
        if english_dictionary_words:
            print(f", Відстань Левенштейна: {similarity_distance}, Найближче слово: {closest_dict_word}")
        else:
            print()