# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Melenoma Detection
# Eeshan Gupta | eeshangpt@gmail.com

# %% [markdown]
# ## Problem Statement
#
# To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

# %% [markdown]
# ## Table of Content
#
# 1. [Problem Statement](#Problem-Statement)
# 2. [Data Reading and Dataset Creation](#Data-Reading-and-Dataset-Creation)
# 3. [Data Visualizations](#Data-Visualizations)
# 4. [Model Building and Training](#Model-Building-and-Training)
#     1. [Building the model]()
#     1. [Training Model]()
#     1. [Visualizing Training Results]()
# 5. [Data Augmentation](#Data-Augmentation)
#     1. [Augmentation Model]()
#     1. [Training Model]()
#     1. [Visualizing Training Results]()
# 6. [Handling Class Imbalance](#Handling-Class-Imbalance)
#     1. [Finding imbalance]()
#     1. [Augmenting dataset]()
#     1. [Creating dataset]()
#     1. [Model building, training and Visualizations]()

# %% [markdown]
# ## Data Reading and Dataset Creation

# %% [markdown]
# **Imports and standard setup**

# %%
# !pip install Augmentor

# %%
from collections import Counter
from glob import glob
from os import getcwd
from os.path import basename, dirname, join
from pathlib import Path

import Augmentor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory

# %%
PRJ_DIR = getcwd()
DATA_DIR = join(PRJ_DIR, 'data')

# %% id="D57L-ovIKtI4"
data_dir_train = Path(join(DATA_DIR, 'Train'))
data_dir_test = Path(join(DATA_DIR, 'Test'))

# %% id="DqksN1w5Fu-N"
image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
print(image_count_train)
image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
print(image_count_test)

# %% [markdown]
# ### Creating Dataset
#
# **Defining loading parameters**

# %%
batch_size = 32
img_height = 180
img_width = 180

# %%
train_ds = image_dataset_from_directory(data_dir_train,
                                        validation_split=0.2,
                                        subset='training',
                                        seed=123,
                                        image_size=(img_height, img_width),
                                        batch_size=batch_size,)

# %%
val_ds = image_dataset_from_directory(data_dir_train,
                                      validation_split=0.2,
                                      subset='validation',
                                      seed=123,
                                      image_size=(img_height, img_width),
                                      batch_size=batch_size,)

# %%
test_ds = image_dataset_from_directory(data_dir_test, seed=123,
                                      image_size=(img_height, img_width),
                                      batch_size=batch_size,)

# %%
class_names = train_ds.class_names
print(class_names)

# %% [markdown]
# ## Data Visualizations

# %% id="tKILZ48I-q1k"
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %% [markdown]
# ## Model Building and Training

# %% [markdown]
# ### Building the model

# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))
])

# %%
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# %%
model.summary()

# %% [markdown]
# ### Model Training

# %%
epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# %% [markdown]
# ### Visualizing Training Results

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# **Conclusion**
#
# - The training accuracy is higher as compared to the validation accuracy
# - The training loss is lower than validation loss
# - The model seems to overfit

# %% [markdown]
# ## Data Augmentation

# %% [markdown]
# ### Augmentation model

# %%
data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal',
                                                 input_shape=(img_height,
                                                              img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),])

# %%
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

# %% [markdown]
# ### Model Building

# %%
model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names)),
])

# %%
model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%
model.summary()

# %%
epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# %% [markdown]
# ### Visualizing the training results

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# **Conclusion**
#
# - The overfitting of the model has decresed and the accuracy and loss for validation almost follows the pattern of training
# - This is an improvement

# %% [markdown]
# ## Handling Class Imbalance
#
# ### Finding the inbalance among classes

# %%
path_list = [x for x in glob(join(data_dir_train, '*', '*.jpg'))]
lesion_list = [basename(dirname(y)) for y in glob(join(data_dir_train, '*', '*.jpg'))]
dataframe_dict_original = dict(zip(path_list, lesion_list))
original_df = pd.DataFrame(list(dataframe_dict_original.items()),columns = ['Path','Label'])
original_df

# %%
X, y = original_df['Path'], original_df['Label']
lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(y)

counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

plt.bar(counter.keys(), counter.values())
plt.show()

# %%
original_df['Label'].value_counts()

# %% [markdown]
# ### Augmenting the datasets

# %%
path_to_training_dataset = join(DATA_DIR, 'Train') + '/'

# %%
for i in class_names:
    p = Augmentor.Pipeline(path_to_training_dataset + i)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.sample(500)

# %%
image_count_train = len(list(data_dir_train.glob('*/output/*.jpg')))
print(image_count_train)

# %%
path_list_new = [x for x in glob(join(data_dir_train, '*','output', '*.jpg'))]
lesion_list_new = [basename(dirname(dirname(y)))
                   for y in glob(join(data_dir_train, '*', 'output', '*.jpg'))]
dataframe_dict_new = dict(zip(path_list_new, lesion_list_new))

df2 = pd.DataFrame(list(dataframe_dict_new.items()),columns = ['Path','Label'])
new_df = original_df.append(df2)
new_df['Label'].value_counts()

# %% [markdown]
# ### Creating the dataset

# %%
train_ds = image_dataset_from_directory(data_dir_train, seed=123,
                                        validation_split = 0.2,
                                        subset = 'training',
                                        image_size=(img_height, img_width),
                                        batch_size=batch_size)

# %%
val_ds = image_dataset_from_directory(data_dir_train, seed=123,
                                      validation_split = 0.2,
                                      subset = 'validation',
                                      image_size=(img_height, img_width),
                                      batch_size=batch_size)

# %%
test_ds = image_dataset_from_directory(data_dir_test,
                                       image_size=(img_height, img_width),
                                       batch_size=batch_size)

# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
# ### Model creation, training, and visualizations

# %%
model = Sequential([layers.experimental.preprocessing.Rescaling(1./255),
                    layers.Conv2D(16, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(32, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(64, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Dropout(0.2),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(len(class_names))
                   ])

# %%
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%
epochs = 20
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs)

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# **Result Analysis**  
# The overfitting reduced. It is evident from the accuracy and loss curves. This is a result of class rebalancing.
