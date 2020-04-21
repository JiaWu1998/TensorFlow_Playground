import tensorflow as tf
from tensorflow import keras
import numpy as np

# Get movie review data set from keras database
data = keras.datasets.imdb

# Seperate data set into training set and testing set. Get the most frequency 10000 words
(train_reviews, train_labels), (test_reviews, test_labels) = data.load_data(num_words = 10000)

# Get the mapping of index and words
# This returns a set of tuples. Here is what the tuple stores: (word,index)
word_index = data.get_word_index()

word_index = {(v+3): k for k, v in word_index.items()}
word_index[0] = "<PAD>"
word_index[1] = "<START>"
word_index[2] = "<UNK>"
word_index[3] = "<UNUSED>"

def decode_review(encoded_review):
    return " ".join([word_index.get(i, "?") for i in encoded_review])

# Normalize the reviews to have 250 words per review
# If review is not long enough, add <PAD> to make it 250 words long
train_reviews = keras.preprocessing.sequence.pad_sequences(train_reviews, value=word_index["<PAD>"], padding="post", maxlen=250)
test_reviews = keras.preprocessing.sequence.pad_sequences(test_reviews, value=word_index["<PAD>"], padding="post", maxlen=250)

# Makes NN model
# Layer 1: Embedding layer. For all 10000 words, have 16 neurons each word. These neurons are the 16 vectors that define the word.
# Layer 2: GlobalAveragePooling layer. 160000 neurons are too much to work with. So I have to average pool it into a 1D layer of 10000 neurons. 
# Layer 3: Hidden layer with 16 neurons. This used the 'relu' activation function. 
# Layer 4: Output layer. Since we only want a binary output(is it a positive or negative review), we only need one neuron. The 'sigmiod' activation function is perfect for this since it maps everything in the range 0 to 1. 

model = keras.Sequential([
    keras.layers.Embedding(10000,16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")
])

# another way to add layers to the model

# model.add(keras.layers.Embedding(10000,16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16,activation="relu"))
# model.add(keras.layers.Dense(1,activation="sigmoid"))

#Compiling the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics = ["accuracy"])

#getting validation data from the testing data set
x_val = train_reviews[:10000]
x_train = train_reviews[10000:]

y_val = train_labels[:10000]
y_label = train_labels[10000:]

#training the model
model.fit(x_train, y_label, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)

#saving the model
model.save("review_mod.h5") 

# evaluate the model
# results = model.evaluate(test_reviews,test_labels)

# Doing sample prediction to see if the model is accuarate 
# predict = model.predict([test_reviews[0].tolist()])
# print(f"\n Review: {decode_review(test_reviews[0])}")
# print(f"\n Actual: {test_labels[0]}")
# print(f"\n Prediction: {predict}")



