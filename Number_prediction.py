# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labes) =  datasets.mnist.load_data()

#Preprocessing: Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to (28, 28, 1) as they are grayscale
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

#Convert the labels to one-hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labes)

#Build the CNN Model
model = models.Sequential()

#First Convolutional layer
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

#Second Convolutional layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#Third Convolutional layer
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#Flatten the 3D output to 1D and add a Dense layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#Output layer with 10 neurons (for 10 digit classes)
model.add(layers.Dense(10, activation='softmax'))

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

#evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

#Make predictions on test images
predictions = model.predict(test_images)
print(f"Prediction for first image: {np.argmax(predictions[6])}")

plt.imshow(test_images[6].reshape(28,28),cmap='gray')
plt.title(f"Predicted Label: {predictions[6].argmax()}")
plt.show()

#predict your own handwriting
from PIL import Image, ImageOps

def prepare_image(path, invert=True):
    img = Image.open(path).convert("L")
    if invert:
        img = ImageOps.invert(img)
    img = img.resize((28,28))
    arr = np.array(img).astype("float32")/255.0
    return arr.reshape(1,28,28,1)

x = prepare_image("image1.png", invert=True)

pred = model.predict(x)
digit = np.argmax(pred)
confidence = np.max(pred) * 100
print(f"Predicted digit: {digit} ({confidence:.1f}% confident")

plt.imshow(x.reshape(28,28), cmap="gray")
plt.title(f"Predicted: {digit}")
plt.axis("off")
plt.show()




























