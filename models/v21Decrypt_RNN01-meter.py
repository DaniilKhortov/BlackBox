import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import Levenshtein # Import the python-Levenshtein library for edit distance

def load_dictionary(file_path="D:/Mysor2/3kurs/Coll_proc_data/Project/EnigmaticCodes.Caesar.csv"):
    #EnigmaticCodes.PL
    #EnigmaticCodes.Atbash
    #EnigmaticCodes.Caesar

    """
    Завантажує словник шифрування з CSV файлу, де дані записані у форматі:
    зашифроване_слово,дешифроване_слово (приклад: mut,nfg).
    """
    dictionary = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Пропускаємо заголовок, якщо він є
            for row in csv_reader:
                if len(row) == 2: # Перевірка, чи рядок містить 2 значення
                    decrypted_word, encrypted_word = row # Порядок у CSV: дешифроване, зашифроване
                    dictionary[encrypted_word.strip()] = decrypted_word.strip() # Записуємо у словник: зашифроване -> дешифроване
                else:
                    print(f"Попередження: Пропущено рядок з некоректною кількістю стовпців: {row}") # Попередження про некоректний рядок
    except FileNotFoundError:
        print(f"Помилка: Файл словника '{file_path}' не знайдено.")
        return None
    except Exception as e:
        print(f"Помилка при читанні CSV файлу '{file_path}': {e}") # Обробка інших можливих помилок при читанні CSV
        return None
    return dictionary

def preprocess_data(dictionary, input_characters=None, target_characters=None):
    """Підготовка даних для RNN: створення словників символів, векторизація, padding."""
    # ... (код preprocess_data без змін)
    encrypted_words = list(dictionary.keys())
    decrypted_words = list(dictionary.values())

    # Створюємо словники символів IF NOT PROVIDED
    if input_characters is None: # Check if input_characters is provided, otherwise calculate
        input_characters = set()
        for word in encrypted_words:
            for char in word:
                input_characters.add(char)
        input_characters = sorted(list(input_characters))
        print(f"preprocess_data (initial) - Calculated input_characters: {input_characters}") # DEBUG PRINT

    if target_characters is None: # Check if target_characters is provided, otherwise calculate
        target_characters = set()
        for word in decrypted_words:
            for char in word:
                target_characters.add(char)
        target_characters = sorted(list(target_characters))
        print(f"preprocess_data (initial) - Calculated target_characters: {target_characters}") # DEBUG PRINT

    # **Explicitly add space to input_characters to ensure padding works**
    if ' ' not in input_characters:
        input_characters = [' '] + input_characters  # Add space at the beginning to maintain sorted order (optional)
        input_characters.sort() # Ensure sorted order after insertion
    if ' ' not in target_characters:
        target_characters = [' '] + target_characters
        target_characters.sort()


    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(word) for word in encrypted_words])
    max_decoder_seq_length = max([len(word) for word in decrypted_words])

    print("preprocess_data - Кількість унікальних вхідних символів:", num_encoder_tokens) # DEBUG PRINT
    print("preprocess_data - Кількість унікальних вихідних символів:", num_decoder_tokens) # DEBUG PRINT
    print("preprocess_data - Максимальна довжина вхідної послідовності:", max_encoder_seq_length)
    print("preprocess_data - Максимальна довжина вихідної послідовності:", max_decoder_seq_length)
    print("preprocess_data - Вхідні символи:", input_characters)
    print("preprocess_data - Вихідні символи:", target_characters)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(encrypted_words), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(decrypted_words), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(decrypted_words), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(dictionary.items()):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        if len(input_text) < max_encoder_seq_length: # Padding only if needed
            encoder_input_data[i, len(input_text):, input_token_index[' ']] = 1.0 # Padding

        for t, char in enumerate(target_text):
            # decoder_target_data випереджає decoder_input_data на один крок у часі
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data буде випереджати на один крок і не включатиме початковий символ.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        if len(target_text) < max_decoder_seq_length: # Padding only if needed
            decoder_input_data[i, len(target_text):, target_token_index[' ']] = 1.0 # Padding
            decoder_target_data[i, len(target_text):, target_token_index[' ']] = 1.0 # Padding


    return encoder_input_data, decoder_input_data, decoder_target_data, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index, input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length

def build_seq2seq_model(num_encoder_tokens, num_decoder_tokens, latent_dim): # Видалено параметр num_layers
    """
    Створює модель Seq2Seq RNN з ОДНИМ шаром LSTM.
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    encoder_lstm = keras.layers.LSTM(latent_dim, return_state=True) # ОДИН LSTM шар
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True) # ОДИН LSTM шар
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Модель, що об'єднує encoder та decoder
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense # Повертаємо ОДИН LSTM шар

def train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=64, epochs=100, validation_split=0.2): # Видалено параметр num_layers
    """
    Навчає модель Seq2Seq.
    """
    print(f"train_model - encoder_input_data.shape: {encoder_input_data.shape}, decoder_input_data.shape: {decoder_input_data.shape}, decoder_target_data.shape: {decoder_target_data.shape}") # DEBUG PRINT
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit( # Store the training history
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
    )
    return model, history # Return the history object


def create_inference_models(model, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense): # Видалено decoder_lstm_layers, decoder_states_inputs, num_layers
    """
    Створює окремі моделі для висновування (encoder_model та decoder_model) для ОДНОШАРОВОЇ RNN.
    """
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = keras.Input(shape=(latent_dim,))
    decoder_state_input_c = keras.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_inputs_inference = keras.Input(shape=(1, num_decoder_tokens)) # Modified shape for inference
    decoder_outputs_inference, state_h_inference, state_c_inference = decoder_lstm( # Використовуємо ОДИН LSTM шар
        decoder_inputs_inference, initial_state=decoder_states_inputs
    )
    decoder_states_inference = [state_h_inference, state_c_inference]
    decoder_outputs_inference = decoder_dense(decoder_outputs_inference)
    decoder_model = keras.Model(
        [decoder_inputs_inference] + decoder_states_inputs, [decoder_outputs_inference] + decoder_states_inference
    )
    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, input_token_index, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens): # Видалено num_layers, latent_dim
    """
    Декодує введену послідовність за допомогою навчених моделей.
    """
    # Кодуємо вхідний стан як вектор станів.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Генеруємо порожню цільову послідовність довжиною 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Заповнюємо перший символ цільової послідовності початковим символом.
    target_seq[0, 0, target_token_index['\t']] = 1.0

    # Цикл дешифрування для генерації символів.
    # Зупиняємося, коли досягаємо граничної умови:
    # 1) Досягнута максимальна довжина вихідної послідовності
    # 2) Знайдено символ зупинки ('.').
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0) # disable progress bar during predict

        # Вибірка токена
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Умова зупинки: або досягнута максимальна довжина, або знайдено символ '\n'
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Оновлюємо цільову послідовність (довжиною 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # **Corrected state update: Use ALL decoder states for the next iteration**
        states_value = [h, c] # Використовуємо стани ОДНОГО LSTM шару

    return decoded_sentence

def calculate_word_similarity(decrypted_word, dictionary_words):
    """
    Визначає схожість дешифрованого слова до справжніх загадувальних слів,
    використовуючи відстань Левенштейна.

    Аргументи:
        decrypted_word (str): Дешифроване слово, яке потрібно оцінити.
        dictionary_words (list): Список справжніх англійських слів (словник).

    Повертає:
        tuple: (min_distance, closest_word)
               min_distance (int): Мінімальна відстань Левенштейна до найближчого слова в словнику.
               closest_word (str): Найближче слово в словнику (з мінімальною відстанню).
    """
    min_distance = float('inf') # Початкове значення для мінімальної відстані (нескінченність)
    closest_word = None

    for dict_word in dictionary_words:
        distance = Levenshtein.distance(decrypted_word, dict_word) # Обчислюємо відстань Левенштейна
        if distance < min_distance: # Якщо знайдено меншу відстань
            min_distance = distance # Оновлюємо мінімальну відстань
            closest_word = dict_word # Запам'ятовуємо найближче слово

    return min_distance, closest_word # Повертаємо мінімальну відстань та найближче слово


if __name__ == "__main__":
    # 1. Завантаження словника
    dictionary = load_dictionary()
    if dictionary is None:
        exit()

    # 2. Підготовка даних (ПЕРШИЙ РАЗ - отримуємо початкові значення, включаючи initial character sets)
    encoder_input_data_initial, decoder_input_data_initial, decoder_target_data_initial, num_encoder_tokens_initial, num_decoder_tokens_initial, input_token_index, target_token_index, input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length = preprocess_data(dictionary)

    # 3. Додаємо спеціальні символи до *існуючих* character sets
    input_characters = [' '] + input_characters # Ensure space is added here as well
    target_characters = [' ', '\t', '\n'] + target_characters # '\t' - start, '\n' - end

    # 4. Оновлюємо словники індексів та кількість токенів ПІСЛЯ додавання спеціальних символів
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    print(f"Before second preprocess_data - num_encoder_tokens: {num_encoder_tokens}, num_decoder_tokens: {num_decoder_tokens}") # DEBUG PRINT

    # 5. **Повторно викликаємо preprocess_data, ПЕРЕДАЮЧИ ОНОВЛЕНІ input_characters та target_characters!**
    encoder_input_data, decoder_input_data, decoder_target_data, _, _, _, _, _, _, _, _ = preprocess_data(dictionary, input_characters=input_characters, target_characters=target_characters) # Pass the updated character sets

    print(f"After second preprocess_data - num_encoder_tokens: {num_encoder_tokens}, num_decoder_tokens: {num_decoder_tokens}") # DEBUG PRINT

     # 6. Побудова моделі Seq2Seq
    latent_dim = 256  # Розмірність прихованого стану LSTM
    model, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense = build_seq2seq_model(num_encoder_tokens, num_decoder_tokens, latent_dim) # Видалено num_layers

    print(f"Before train_model - num_encoder_tokens: {num_encoder_tokens}, num_decoder_tokens: {num_decoder_tokens}") # DEBUG PRINT

    # 7. Навчання моделі
    epochs = 100 # Increased epochs
    batch_size = 128 # Increased batch_size (you can try different values)
    model, training_history = train_model(model, encoder_input_data, decoder_input_data, decoder_target_data, epochs=epochs, batch_size=batch_size) # Видалено num_layers та latent_dim

    # 8. Створення моделей для висновування
    encoder_model, decoder_model = create_inference_models(model, encoder_inputs, encoder_states, decoder_inputs, decoder_lstm, decoder_dense) # Видалено decoder_lstm_layers, decoder_states_inputs, num_layers
    # 9. Побудова графіків навчання
    #import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('Втрати моделі')
    plt.ylabel('Втрати')
    plt.xlabel('Епоха')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # 10. Завантаження списку "справжніх" англійських слів (словник)
    with open("D:/Mysor2/3kurs/Coll_proc_data/Project/words_alphaC.txt", 'r') as word_file: # Замініть на шлях до вашого словника
        #words_alphaA
        #words_alphaPL
        #
        english_dictionary_words = [word.strip() for word in word_file]

    # 11. Тестування дешифрування
    input_words = list(dictionary.keys())
    num_test_words = 10
    test_words = input_words[:num_test_words]

    for seq_index in range(num_test_words):
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, input_token_index, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens)
        input_word = test_words[seq_index]
        expected_decoding = dictionary[input_word]
        similarity_distance, closest_dict_word = calculate_word_similarity(decoded_sentence.strip(), english_dictionary_words) # Оцінюємо схожість

        print('-')
        print('Зашифроване слово:', input_word)
        print('Дешифроване слово (RNN):', decoded_sentence.strip())
        print('Очікуване дешифрування:', expected_decoding)
        print(f"Відстань Левенштейна до найближчого слова: {similarity_distance}, Найближче слово: {closest_dict_word}") # Виводимо схожість

    # 12. Дешифрування невідомих слів (приклад)
    unknown_encrypted_words = ["ergzorp", "wzmrro", "rihosre"] # vitalik, daniil, irslhiv 
    # "ergzorp", "wzmrro", "rihosre" - atbash # vitalik, daniil, irslhiv 
    # "italikvay", "aniilday", "irslhivyay" - PL # vitalik, daniil, irslhiv 

    # "dedqgrq", "delolwb", "deoh" - Caesar #3 #abandon ability able
    # "nonaqba", "novyvgl", "noyr" - Caesar #13 #abandon ability able
    # "xyxkalk", "xyfifqv", "xyib" - Caesar #23 #abandon ability able
    print("\nДешифрування невідомих слів:")
    for encrypted_word in unknown_encrypted_words:
        test_input_data = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        for t, char in enumerate(encrypted_word):
            if char in input_token_index: # Перевірка на невідомі символи
                test_input_data[0, t, input_token_index[char]] = 1.0
        test_input_data[0, len(encrypted_word):, input_token_index[' ']] = 1.0 # Padding

        decoded_sentence = decode_sequence(test_input_data, encoder_model, decoder_model, input_token_index, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens)
        similarity_distance, closest_dict_word = calculate_word_similarity(decoded_sentence.strip(), english_dictionary_words) # Оцінюємо схожість

        print(f"Зашифроване слово: {encrypted_word}, Дешифроване слово (RNN): {decoded_sentence.strip()}, Відстань Левенштейна: {similarity_distance}, Найближче слово: {closest_dict_word}") # Виводимо схожість