from google.colab import drive
drive.mount(’/content/drive’)

!unzip ’/content/drive/MyDrive/Dataset/Cardamom_Plant_Dataset.zip’

!pip install scikit-learn==1.0.0

import sklearn
print(sklearn._version_)

import numpy as np
import pandas as pd
import os
import pathlib
import tensorflow as tf
import seaborn as sn
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model, image_dataset_from_dir
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_m
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropo

data_dir = pathlib.Path("/content/Cardamom_Plant_Dataset/Cardamom_Plant_Dataset")
data = image_dataset_from_directory(data_dir, seed = 123, image_size=(224, 224))
class_names = data.class_names

labels = np.concatenate([y for x,y in data], axis=0)
values = pd.value_counts(labels)
values.plot(kind=’bar’)

plt.figure(figsize=(10, 10))
for images, labels in data.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(data.class_names[labels[i]])
    plt.axis("off")
    
data = data.map(lambda x, y: (x/255, y))
train_size = int(0.7 * len(data)) +1
val_size = int(0.2 * len(data))
test_size = int(0.1 * len(data))
train = data.take(train_size)
remaining = data.skip(train_size)
val = remaining.take(val_size)
test = remaining.skip(val_size)

test_iter = test.as_numpy_iterator()
test_set = {"images":np.empty((0,224,224,3)), "labels":np.empty(0)}
while True:
  try:
    batch = test_iter.next()
    test_set[’images’] = np.concatenate((test_set[’images’], batch[0]))
    test_set[’labels’] = np.concatenate((test_set[’labels’], batch[1]))
  except:
    break
y_test = test_set[’labels’]

def evaluate_model(model):
  model.evaluate(test)
  y_pred = np.argmax(model.predict(test_set[’images’]), 1)
  print(classification_report(y_test, y_pred, target_names = class_names))
  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(10, 8))
  sn.heatmap(cm, annot=True)
  plt.xticks(np.arange(3)+.5, class_names, rotation=90)
  plt.yticks(np.arange(3)+.5, class_names)
  plt.xlabel("Predicted")
  plt.ylabel("Actual")
  plt.title("Confusion Matrix")
  
model = Sequential([
        Conv2D(64, (3,3), activation = ’relu’,padding=’same’, input_shape=(224,224,3)
        Conv2D(64, (3,3), activation = ’relu’,padding=’same’),
        Conv2D(64, (3,3), activation = ’relu’,padding=’same’),
        MaxPool2D(),
        Conv2D(128, (3,3), padding=’same’, activation = ’relu’),
        Conv2D(128, (3,3),padding=’same’, activation = ’relu’),
        Conv2D(128, (3,3), activation = ’relu’,padding=’same’),
        MaxPool2D(),
        Flatten(),
        Dense(256, activation = ’relu’),
        Dense(4, activation=’softmax’)
      ])

model.compile(loss=’sparse_categorical_crossentropy’, optimizer=’adam’, metrics=[’acc
model.summary()
plot_model(model, to_file=’simple-cnn.png’, show_shapes=True)
history = model.fit(train, validation_data=val, epochs = 30)

def plot_performance(epochs, history):
  acc = history.history[’accuracy’]
  val_acc = history.history[’val_accuracy’]
  loss = history.history[’loss’]
  val_loss = history.history[’val_loss’]
  epochs_range = range(epochs)
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label=’Training Accuracy’)
  plt.plot(epochs_range, val_acc, label=’Validation Accuracy’)
  plt.legend(loc=’lower right’)
  plt.title(’Training and Validation Accuracy’)
  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label=’Training Loss’)
  plt.plot(epochs_range, val_loss, label=’Validation Loss’)
  plt.legend(loc=’upper right’)
  plt.title(’Training and Validation Loss’)
  plt.show()

plot_performance(30, history)
evaluate_model(model)
model.save("cardamom.h5")
pip install tensorflowjs
!pip install tensorflowjs
!tensorflowjs_converter \
    --input_format=keras \
    cardamom.h5 \
    /tmp/my_tfjs_model
              
model = tf.keras.models.load_model("/content/cardamom.h5")
CLASSES = [ ’Healthy’, ’LeafBlight’, ’LeafSpot’ ]

img_path = ’/content/Cardamom_Plant_Dataset/Cardamom_Plant_Dataset/LeafBlight/C_Blight.jpg
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Make a prediction on the preprocessed image
preds = model.predict(x)
predicted_class = np.argmax(preds)

# Print the predicted class label
print("Predicted class: ", CLASSES[predicted_class])
