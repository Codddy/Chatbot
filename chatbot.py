import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, RepeatVector, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Veri yükleme
with open('human_text.txt', 'r', encoding='utf-8') as file:
    human_texts = file.read().splitlines()

with open('robot_text.txt', 'r', encoding='utf-8') as file:
    robot_texts = file.read().splitlines()

# Giriş ve çıkış metinlerini tokenlara dönüştürme
tokenizer_input = Tokenizer()
tokenizer_input.fit_on_texts(human_texts)
input_sequences = tokenizer_input.texts_to_sequences(human_texts)
max_len_input = max(len(seq) for seq in input_sequences)
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len_input)

tokenizer_output = Tokenizer()
tokenizer_output.fit_on_texts(robot_texts)
output_sequences = tokenizer_output.texts_to_sequences(robot_texts)
max_len_output = max(len(seq) for seq in output_sequences)
padded_output_sequences = pad_sequences(output_sequences, maxlen=max_len_output, padding='post')

# Model parametreleri
input_vocab_size = len(tokenizer_input.word_index) + 1
output_vocab_size = len(tokenizer_output.word_index) + 1
embedding_dim = 50
latent_dim = 100

# Model oluşturma
model = Sequential()
model.add(Embedding(input_vocab_size, embedding_dim, input_length=max_len_input, mask_zero=True))
model.add(LSTM(latent_dim))
model.add(RepeatVector(max_len_output))
model.add(LSTM(latent_dim, return_sequences=True))
model.add(Dense(output_vocab_size, activation='softmax'))

# Model derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model eğitimi
model.fit(padded_input_sequences, np.expand_dims(padded_output_sequences, axis=-1), epochs=1000)

# Cevap üretme fonksiyonu
def generate_response(input_text):
    input_seq = tokenizer_input.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_len_input)
    pred = model.predict(input_seq)
    pred = np.argmax(pred, axis=-1)[0]
    output_text = ' '.join(tokenizer_output.index_word[idx] for idx in pred if idx != 0)
    return output_text

# Sohbet başlatma
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = generate_response(user_input)
    print(f'Chatbot: {response}')
