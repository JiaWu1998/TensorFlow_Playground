import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_reviews, train_labels), (test_reviews, test_labels) = data.load_data(num_words = 10000)

word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reversed_word_index = {v: k for k, v in word_index.items()}

def decode_review(encoded_review):
    return " ".join([reversed_word_index.get(i, "?") for i in encoded_review])

train_reviews = keras.preprocessing.sequence.pad_sequences(train_reviews, value=word_index["<PAD>"], padding="post", maxlen=250)
test_reviews = keras.preprocessing.sequence.pad_sequences(test_reviews, value=word_index["<PAD>"], padding="post", maxlen=250)

model = keras.Sequential([
    keras.layers.Embedding(10000,16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")
])
# model.add(keras.layers.Embedding(10000,16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16,activation="relu"))
# model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

#validation data
x_val = train_reviews[:10000]
x_train = train_reviews[10000:]

y_val = train_labels[:10000]
y_label = train_labels[10000:]

fit_model = model.fit(x_train, y_label, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)

model.save("review_mod.h5") 
# results = model.evaluate(test_reviews,test_labels)

# predict = model.predict([test_reviews[0].tolist()])
# print(f"\n Review: {decode_review(test_reviews[0])}")
# print(f"\n Actual: {test_labels[0]}")
# print(f"\n Prediction: {predict}")



