import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers
from keras.datasets import imdb
import os
from config import vectorize_sequences


BASE_DIR = os.path.dirname(__file__)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# κανουμε vectorize τα δεδομενα
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # sigmoid για να βγαλει αποτελεσμα μεταξυ 2 κλασεων μονο
# για την  ακριβεια την πιθανοτητα να ειναι μεταξυ των 2 κλασεων

# model.compile(
#     optimizer=optimizers.RMSprop(lr=0.001),
#     loss=losses.binary_crossentropy,
#     metrics=[metrics.binary_accuracy]
# )

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(partial_x_train,
					partial_y_train,
					epochs=20,
					batch_size=512,
					validation_data=(x_val, y_val)
					)
history_dict = history.history
loss_values = history_dict.get('loss')
val_loss_values = history_dict.get('val_loss')
model.save(os.path.join(BASE_DIR, 'models/imdb_classification.h5'))

# epochs = range(1, len(acc) + 1)
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
