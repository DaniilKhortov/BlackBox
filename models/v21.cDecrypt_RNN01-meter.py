import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import Levenshtein # Потрібно встановити: pip install python-Levenshtein

# --- Оновлена функція завантаження даних для Цезаря ---
def load_dictionary(file_path="D:/Mysor2/3kurs/Coll_proc_data/Project/data_r.csv"):
    #EnigmaticCodes.Caesar
    """
    Завантажує словник шифру Цезаря з CSV файлу.
    Очікує формат: Original,Slide,Encripted
    Повертає два словники:
        - dictionary: {зашифроване_слово: дешифроване_слово}
        - slide_map: {зашифроване_слово: зсув}
    """
    dictionary = {}
    slide_map = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader) # Читаємо заголовок

            try:
                original_col_index = header.index("Original")
                slide_col_index = header.index("Slide")
                encrypted_col_index = header.index("Encripted")
            except ValueError:
                print(f"Помилка: Не знайдено необхідні стовпці ('Original', 'Slide', 'Encripted') у заголовку файлу {file_path}")
                return None, None

            for row in csv_reader:
                if len(row) >= 3: # Перевірка, чи рядок містить щонайменше 3 значення
                    try:
                        decrypted_word = row[original_col_index].strip()
                        slide = int(row[slide_col_index].strip())
                        encrypted_word = row[encrypted_col_index].strip()

                        dictionary[encrypted_word] = decrypted_word
                        slide_map[encrypted_word] = slide
                    except (ValueError, IndexError) as e:
                         print(f"Попередження: Помилка при обробці рядка: {row} у файлі {file_path}. Помилка: {e}")
                else:
                    print(f"Попередження: Пропущено рядок з некоректною кількістю стовпців: {row}")
    except FileNotFoundError:
        print(f"Помилка: Файл словника '{file_path}' не знайдено.")
        return None, None
    except Exception as e:
        print(f"Помилка при читанні CSV файлу '{file_path}': {e}")
        return None, None
    return dictionary, slide_map

# --- Оновлена функція підготовки даних ---
def preprocess_data(dictionary, slide_map):
    """
    Підготовка даних для RNN моделі Цезаря:
    Створення словників символів, векторизація слів та зсувів, padding.
    """
    encrypted_words = list(dictionary.keys())
    decrypted_words = list(dictionary.values())
    slide_values = [slide_map[word] for word in encrypted_words] # Список зсувів

    # Створення словників символів
    input_characters = set()
    target_characters = set()
    for word in encrypted_words:
        for char in word:
            input_characters.add(char)
    for word in decrypted_words:
        for char in word:
            target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))

    # Додаємо спеціальні символи
    if ' ' not in input_characters:
        input_characters = [' '] + input_characters
        input_characters.sort()
    if ' ' not in target_characters:
        target_characters = [' '] + target_characters
        target_characters.sort()
    if '\t' not in target_characters: # Додаємо символ початку послідовності
        target_characters = ['\t'] + target_characters
        target_characters.sort()
    if '\n' not in target_characters: # Додаємо символ кінця послідовності
        target_characters = ['\n'] + target_characters
        target_characters.sort()


    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(word) for word in encrypted_words])
    max_decoder_seq_length = max([len(word) for word in decrypted_words]) + 1 # +1 для токенів початку/кінця

    print("Кількість унікальних вхідних символів:", num_encoder_tokens)
    print("Кількість унікальних вихідних символів:", num_decoder_tokens)
    print("Макс. довжина вхідної послідовності:", max_encoder_seq_length)
    print("Макс. довжина вихідної послідовності:", max_decoder_seq_length)
    # print("Вхідні символи:", input_characters)
    # print("Вихідні символи:", target_characters)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    # Векторизація даних
    encoder_input_data = np.zeros(
        (len(encrypted_words), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(decrypted_words), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(decrypted_words), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    slide_input_data = np.array(slide_values, dtype="float32").reshape(-1, 1) # Масив зсувів

    for i, (input_text, target_text) in enumerate(dictionary.items()):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, len(input_text):, input_token_index[' ']] = 1.0 # Padding

        # Додаємо токен початку до decoder_input і target_text
        target_text_with_tokens = '\t' + target_text + '\n'
        for t, char in enumerate(target_text_with_tokens):
            if t < max_decoder_seq_length: # Обмежуємо довжину
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data зсунутий на один крок і не включає '\t'
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, len(target_text_with_tokens):, target_token_index[' ']] = 1.0 # Padding
        decoder_target_data[i, len(target_text_with_tokens)-1:, target_token_index[' ']] = 1.0 # Padding


    return (encoder_input_data, decoder_input_data, slide_input_data, decoder_target_data,
            num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index,
            input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length)

# --- Оновлена функція побудови моделі з урахуванням зсуву ---
def build_seq2seq_model(num_encoder_tokens, num_decoder_tokens, latent_dim):
    """
    Створює модель Seq2Seq RNN для Цезаря з додатковим входом для зсуву.
    """
    # Encoder
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens), name='encoder_input')
    encoder_lstm = keras.layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Slide Input
    slide_input = keras.Input(shape=(1,), name='slide_input')
    # Можна додати Embedding для зсуву, якщо зсувів багато і вони категоріальні
    # slide_embedding_dim = 16
    # slide_embedding_layer = keras.layers.Embedding(input_dim=27, output_dim=slide_embedding_dim, name='slide_embedding')(slide_input)
    # slide_embedding_layer = keras.layers.Flatten()(slide_embedding_layer) # Розгортаємо Embedding

    # Або просто Dense шар для числового зсуву
    slide_dense = keras.layers.Dense(latent_dim, activation='relu', name='slide_dense')(slide_input)

    # Об'єднуємо стани енкодера зі зсувом (приклад: конкатенація)
    combined_state_h = keras.layers.Concatenate(axis=-1, name='concat_h')([state_h, slide_dense])
    combined_state_c = keras.layers.Concatenate(axis=-1, name='concat_c')([state_c, slide_dense])

    # Адаптуємо розмірність об'єднаних станів до розмірності LSTM декодера
    # Використовуємо Dense шари для проекції назад до latent_dim
    decoder_state_h_input = keras.layers.Dense(latent_dim, activation='tanh', name='decoder_state_h_proj')(combined_state_h)
    decoder_state_c_input = keras.layers.Dense(latent_dim, activation='tanh', name='decoder_state_c_proj')(combined_state_c)
    decoder_initial_states = [decoder_state_h_input, decoder_state_c_input]

    # Decoder
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens), name='decoder_input')
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=decoder_initial_states) # Передаємо адаптовані стани
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax", name='decoder_output_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Модель, що об'єднує encoder, slide_input та decoder
    model = keras.Model([encoder_inputs, decoder_inputs, slide_input], decoder_outputs) # Додаємо slide_input до входів моделі
    return model

# --- Оновлена функція навчання ---
def train_model(model, encoder_input_data, decoder_input_data, slide_input_data, decoder_target_data, batch_size=64, epochs=100, validation_split=0.2):
    """Навчає модель Seq2Seq для Цезаря."""
    print(f"train_model shapes - encoder_input: {encoder_input_data.shape}, decoder_input: {decoder_input_data.shape}, slide_input: {slide_input_data.shape}, decoder_target: {decoder_target_data.shape}")
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        [encoder_input_data, decoder_input_data, slide_input_data], # Передаємо дані про зсув
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
    )
    return model, history

# --- Оновлені функції для висновування ---
def create_inference_models(model):
    """Створює моделі для висновування для моделі Цезаря."""
    encoder_inputs = model.input[0]  # encoder_inputs
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer('encoder_lstm').output # Вихід та стани енкодера
    encoder_states = [state_h_enc, state_c_enc]

    slide_input_inf = model.input[2] # slide_input
    slide_dense_inf = model.get_layer('slide_dense')(slide_input_inf) # Обробка зсуву

    # Об'єднуємо стани енкодера зі зсувом для висновування
    combined_state_h_inf = model.get_layer('concat_h')([state_h_enc, slide_dense_inf])
    combined_state_c_inf = model.get_layer('concat_c')([state_c_enc, slide_dense_inf])
    decoder_state_h_input_inf = model.get_layer('decoder_state_h_proj')(combined_state_h_inf)
    decoder_state_c_input_inf = model.get_layer('decoder_state_c_proj')(combined_state_c_inf)
    encoder_states_inf = [decoder_state_h_input_inf, decoder_state_c_input_inf] # Адаптовані стани для декодера

    encoder_model = keras.Model([encoder_inputs, slide_input_inf], encoder_states_inf) # Енкодер тепер приймає і зсув

    decoder_inputs_inf = model.input[1]  # decoder_inputs
    decoder_state_input_h = keras.Input(shape=(latent_dim,), name='inf_decoder_state_h')
    decoder_state_input_c = keras.Input(shape=(latent_dim,), name='inf_decoder_state_c')
    decoder_states_inputs_inf = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs_inf, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs_inf, initial_state=decoder_states_inputs_inf
    )
    decoder_states_inf = [state_h_dec, state_c_dec]
    decoder_dense = model.get_layer('decoder_output_dense')
    decoder_outputs_inf = decoder_dense(decoder_outputs_inf)

    decoder_model = keras.Model(
        [decoder_inputs_inf] + decoder_states_inputs_inf,
        [decoder_outputs_inf] + decoder_states_inf
    )

    return encoder_model, decoder_model

def decode_sequence(input_seq, slide_value, encoder_model, decoder_model, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens):
    """Декодує введену послідовність для Цезаря, використовуючи зсув."""
    slide_input_inf = np.array([[slide_value]], dtype='float32') # Підготовка входу для зсуву

    # Кодуємо вхідний стан як вектор станів, передаючи зсув
    states_value = encoder_model.predict([input_seq, slide_input_inf], verbose=0)

    # Генеруємо порожню цільову послідовність довжиною 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Заповнюємо перший символ цільової послідовності початковим символом.
    target_seq[0, 0, target_token_index['\t']] = 1.0

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return decoded_sentence

# --- Функція схожості (без змін) ---
def calculate_word_similarity(decrypted_word, dictionary_words):
    """Визначає схожість дешифрованого слова."""
    # ... (код calculate_word_similarity без змін)
    min_distance = float('inf')
    closest_word = None
    for dict_word in dictionary_words:
        distance = Levenshtein.distance(decrypted_word, dict_word)
        if distance < min_distance:
            min_distance = distance
            closest_word = dict_word
    return min_distance, closest_word

# --- Головний блок ---
if __name__ == "__main__":
    # 1. Завантаження словника Цезаря
    dictionary, slide_map = load_dictionary()
    if dictionary is None:
        exit()

    # 2. Підготовка даних для Цезаря
    (encoder_input_data, decoder_input_data, slide_input_data, decoder_target_data,
     num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index,
     input_characters, target_characters, max_encoder_seq_length, max_decoder_seq_length) = preprocess_data(dictionary, slide_map)

    # Створення зворотного словника для вихідних символів
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

     # 3. Побудова моделі Seq2Seq для Цезаря
    latent_dim = 256
    model = build_seq2seq_model(num_encoder_tokens, num_decoder_tokens, latent_dim)
    model.summary() # Виводимо структуру моделі
    keras.utils.plot_model(model, to_file='caesar_seq2seq_model.png', show_shapes=True) # Візуалізація моделі

    # 4. Навчання моделі для Цезаря
    epochs = 100
    batch_size = 128
    model, training_history = train_model(model, encoder_input_data, decoder_input_data, slide_input_data, decoder_target_data, epochs=epochs, batch_size=batch_size)

    # 5. Створення моделей для висновування
    encoder_model_inf, decoder_model_inf = create_inference_models(model)

    # 6. Побудова графіків навчання
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('Точність моделі (Цезар)')
    plt.ylabel('Точність')
    plt.xlabel('Епоха')
    plt.legend(['Навчання', 'Валідація'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('Втрати моделі (Цезар)')
    plt.ylabel('Втрати')
    plt.xlabel('Епоха')
    plt.legend(['Навчання', 'Валідація'], loc='upper left')
    plt.show()

    # 7. Завантаження списку "справжніх" англійських слів (словник)
    try:
        with open("D:/Mysor2/3kurs/Coll_proc_data/Project/words_alphaC_r.txt", 'r') as word_file: # Використовуємо спільний словник
            english_dictionary_words = [word.strip() for word in word_file]
    except FileNotFoundError:
        print("Помилка: Файл словника англійських слів 'words_alpha.txt' не знайдено.")
        english_dictionary_words = [] # Створюємо порожній список, щоб уникнути помилок далі

    # 8. Тестування дешифрування (слова зі словника навчання)
    print("\n--- Тестування дешифрування (слова зі словника навчання) ---")
    input_words_list = list(dictionary.keys())
    num_test_words = 10
    for seq_index in range(min(num_test_words, len(encoder_input_data))): # Беремо перші num_test_words або менше, якщо даних мало
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        input_word = input_words_list[seq_index]
        slide_value = slide_map[input_word] # Отримуємо зсув для цього слова
        expected_decoding = dictionary[input_word]

        decoded_sentence = decode_sequence(input_seq, slide_value, encoder_model_inf, decoder_model_inf, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens)
        similarity_distance, closest_dict_word = calculate_word_similarity(decoded_sentence.strip(), english_dictionary_words)

        print('-')
        print(f"Зашифроване слово: {input_word} (Зсув: {slide_value})")
        print(f"Дешифроване слово (RNN): {decoded_sentence.strip()}")
        print(f"Очікуване дешифрування: {expected_decoding}")
        if english_dictionary_words:
             print(f"Відстань Левенштейна до найближчого слова: {similarity_distance}, Найближче слово: {closest_dict_word}")

    # 9. Дешифрування невідомих слів (приклад)
    print("\n--- Дешифрування невідомих слів ---")
    unknown_encrypted_words_with_slides = {
        "dedqgrq": 3, #abandon
        "delolwb": 3, #ability
        "deoh": 3,    #able

        "cdcpfqp": 2, #abandon
        "cdknkva": 2, #ability
        "cdng": 2,    #able
  
        "eferhsr": 4, #abandon
        "efmpmxc": 4, #ability
        "efpi": 4     #able


    }
       # "dedqgrq": 3, #abandon
       # "delolwb": 3, #ability
       # "deoh": 3,    #able
  
       # "nonaqba": 13, #abandon
       # "novyvgl": 13, #ability
       # "noyr": 13,    #able
  
       # "xyxkalk": 23, #abandon
       # "xyfifqv": 23, #ability
       # "xyib": 23,    #able
  
       # "uryyb": 13, # hello
       # "jbeyq": 13, # world
    for encrypted_word, slide_value in unknown_encrypted_words_with_slides.items():
        test_input_data = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        for t, char in enumerate(encrypted_word):
            if t < max_encoder_seq_length and char in input_token_index: # Обмежуємо довжину та перевіряємо наявність символу
                test_input_data[0, t, input_token_index[char]] = 1.0
        test_input_data[0, len(encrypted_word):, input_token_index[' ']] = 1.0 # Padding

        decoded_sentence = decode_sequence(test_input_data, slide_value, encoder_model_inf, decoder_model_inf, target_token_index, reverse_target_char_index, max_decoder_seq_length, num_decoder_tokens)
        similarity_distance, closest_dict_word = calculate_word_similarity(decoded_sentence.strip(), english_dictionary_words)

        print(f"Зашифроване слово: {encrypted_word} (Зсув: {slide_value}), Дешифроване слово (RNN): {decoded_sentence.strip()}", end="")
        if english_dictionary_words:
            print(f", Відстань Левенштейна: {similarity_distance}, Найближче слово: {closest_dict_word}")
        else:
            print() # Просто новий рядок, якщо немає словника для порівняння