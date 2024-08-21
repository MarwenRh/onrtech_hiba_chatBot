import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
import string
from googletrans import Translator
import random

data_file = open('chatbot.json', encoding="utf8").read()
intents = json.loads(data_file)
patterns = []
tags = []
responses = {}
for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['patterns']:
        patterns.append(line)
        tags.append(intent['tag'])
data = {'patterns': patterns, 'tags': tags}
data_df = pd.DataFrame(data)
data_df['patterns'] = data_df['patterns'].apply(lambda x: x.lower())
data_df['patterns'] = data_df['patterns'].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data_df['patterns'])
train_sequences = tokenizer.texts_to_sequences(data_df['patterns'])
train_pad = pad_sequences(train_sequences)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(data_df['tags'])
input_shape = train_pad.shape[1]
vocabulary_size = len(tokenizer.word_index) + 1
output_length = len(set(train_labels))
text_input = Input(shape=(input_shape,))
text_embedding = Embedding(input_dim=vocabulary_size, output_dim=200)(text_input)
text_lstm = LSTM(200, return_sequences=True)(text_embedding)
text_flatten = Flatten()(text_lstm)
output = Dense(output_length, activation='softmax')(text_flatten)
model = Model(inputs=text_input, outputs=output)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(train_pad, train_labels, epochs=300)
model.save('bot.keras')
model.save('bot.h5')
while True:
    prediction_input = input('You: ')
    translator = Translator()
    print("Prediction input:", prediction_input)
    translation = translator.translate(prediction_input, dest='en')
    language_code = translation.src
    prediction_input = translation.text.lower().translate(str.maketrans('', '', string.punctuation))
    prediction_input_seq = tokenizer.texts_to_sequences([prediction_input])
    prediction_input_pad = pad_sequences(prediction_input_seq, maxlen=input_shape)
    output = model.predict(prediction_input_pad)
    predicted_tag = label_encoder.inverse_transform([np.argmax(output)])[0]
    response = random.choice(responses[predicted_tag])
    translated_response = translator.translate(response, dest=language_code).text
    print('Bot:', translated_response)
    if predicted_tag == "goodbye":
        break