import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

# Get fashion data set from keras database
data = keras.datasets.fashion_mnist

# Seperate data into training sets and testing sets
# All the images are 2D array that is 28 x 28 because the image is 28 pixels wide and 28 pixels tall
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# The fashion data set has labels all 10 clothing with integers 0-9
# This is the mapping from the integer value(index of the array) to the name of the clothing
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker', 'Bag', 'Ankle boot']

# All images have black pixel value of 0 to 255. 
# We can normalizing the range of 0-255 to 0-1 so the data is easier to work with 
train_images = train_images/255.0
test_images = test_images/255.0 

# Building the neural network model
# Layer 1: Flatten input layer with 28 x 28 neurons. Each neuron holds the pixel value of one pixel in the image
# Layer 2: Hidden layer with 128 neurons. It uses the 'relu' activation function
# Layer 3: Output layer with 10 neurons. Each neuron represents the one of the clothing. It uses the 'softmax' activation function 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compiling the NN model 
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Training the model with the training data set
model.fit(train_images, train_labels, epochs=5)

# Evaluating the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Doing 10 predictions using the model 
predictions = model.predict(test_images)

for i in range(10):
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.title(f"Prediction: {class_names[np.argmax(predictions[i])]}")
    plt.xlabel(f"Actual: {class_names[test_labels[i]]}")
    plt.show()

# Saving the model in the fashion_model.h5 file
model.save("fashion_model.h5")

