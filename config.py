import numpy as np


# κανει τα test & train data 1 και 0 για να θεωρουνται tensors. πρεπει να γινει για να
# μπορουμε να τα εισαγουμε στο μοντελο
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))

	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1

	return results


# categorical encoding. χρησιμοποιειται συχνα για vectorization catagorical δεδομενων. μπορει να γινει και με την
# built-in μεθοδο του keras to_categorical
def to_one_hot(labels, dimension=46):
	results = np.zeros(len(labels), dimension)

	for i, label in enumerate(labels):
		results[i, label] = 1

	return results
