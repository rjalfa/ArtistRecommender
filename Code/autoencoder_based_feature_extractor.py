import sys

import keras.backend as K
import numpy as np
import numpy.random as rnd
import tensorflow as tf

from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model

from sklearn.model_selection import KFold, train_test_split

import get_data

class AutoEncoderFeatureExtractor:

	def get_matrix_from_list(self, rows, columns, ratings, total_rows, total_columns):
		ratings_matrix = [[0. for i in range(total_columns)] for j in range(total_rows)]

		for row_id, column_id, val in zip(rows, columns, ratings):
			# print(row_id, column_id, val, total_rows, total_columns)
			ratings_matrix[row_id][column_id] = min(val, 300)

		return np.array(ratings_matrix).astype(float)

	def train_model_from_list(self, rows, columns, ratings, total_rows, total_columns, num_hidden_nodes, regularizer, epoch=50, batch_size=256):
		ratings_matrix = self.get_matrix_from_list(rows, columns, ratings, total_rows, total_columns)

		self.train_model_from_matrix(ratings_matrix, num_hidden_nodes, regularizer, epoch, batch_size)

	def calculate_biases(self):
		self.global_average = np.sum(self.ratings_matrix) / np.count_nonzero(self.rated_mask)

		cnt_non_zero = np.count_nonzero(self.rated_mask, axis = 0)
		cnt_non_zero[cnt_non_zero == 0] = 1
		self.column_biases = self.global_average - (np.sum(self.ratings_matrix, axis = 0) / cnt_non_zero)

		cnt_non_zero = np.count_nonzero(self.rated_mask, axis = 1)
		cnt_non_zero[cnt_non_zero == 0] = 1
		self.row_biases = self.global_average - (np.sum(self.ratings_matrix, axis = 1) / cnt_non_zero)

		self.ratings_matrix -= self.global_average + self.column_biases + self.row_biases.reshape(self.row_biases.shape[0], 1)
		self.ratings_matrix[self.rated_mask == 0] = 0

	def train_model_from_matrix(self, ratings_matrix, num_hidden_nodes, regularizer, epoch, batch_size):
		self.ratings_matrix = ratings_matrix

		self.rated_mask = (self.ratings_matrix != 0)

		self.calculate_biases()
		self.autoencoder = self.generate_autoencoder(num_hidden_nodes, regularizer, epoch, batch_size)

		self.predicted_ratings_matrix = self.predict_ratings_matrix()

	def generate_autoencoder(self, num_hidden_nodes, regularizer, epoch, batch_size):
		def masked_loss(y_true, y_pred):
			non_zeros = tf.not_equal(y_true, tf.constant(0.0))
			masked_y_pred = tf.where(non_zeros, y_pred, y_true)

			return K.sum(K.square(masked_y_pred - y_true)) / tf.to_float(tf.count_nonzero(y_true))

		encoding_dim = num_hidden_nodes
		input_dim = self.ratings_matrix.shape[1]

		main_input = Input(shape=(input_dim,), dtype='float32')
		encoded = Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l2(regularizer))(main_input)
		decoded = Dense(input_dim, activation='linear', kernel_regularizer=regularizers.l2(regularizer))(encoded)

		autoencoder = Model(main_input, decoded)
		# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		autoencoder.compile(optimizer='adagrad', loss=masked_loss, metrics=['mae', 'mse'])
		# autoencoder.compile(optimizer='nadam', loss=masked_loss, metrics=['mae'])
		autoencoder.fit(self.ratings_matrix, self.ratings_matrix, epochs=epoch, batch_size=batch_size, verbose=1)

		return autoencoder

	def predict_ratings_matrix(self):
		predicted_ratings = self.autoencoder.predict(self.ratings_matrix)
		predicted_ratings += self.global_average + self.column_biases + self.row_biases.reshape(self.row_biases.shape[0], 1)
		predicted_ratings = (1 - self.rated_mask) * predicted_ratings

		predicted_ratings[predicted_ratings < 0] = 0
		predicted_ratings[predicted_ratings > 300] = 300

		return predicted_ratings

	# def calculate_nmae_error(self, target_ratings):
	# 	return np.mean(np.absolute(self.predicted_ratings - target_ratings)) / 100

	# def predict(self, rid, cid):
	# 	return self.predicted_ratings[rid][cid]

	# def test_model_from_file(self, file_name):
	# 	users, items, ratings = self.get_user_item_list(file_name)
	# 	return self.test_model(users, items, ratings)

	# def test_model(self, users, items, ratings):
	# 	vpredict = np.vectorize(lambda rid, cid: self.predict(uid, iid))

	# 	predictions = vpredict(items, users) if self.item_based else vpredict(users, items)
		
	# 	return np.mean(np.absolute(predictions - np.array(ratings))) / 4

if __name__ == "__main__":
	rnd.seed(42)

	users, artists, playcounts, total_users, total_artists = get_data.get_data_as_list('filtered_data.tsv')
	users_train, users_test, artists_train, artists_test, playcounts_train, playcounts_test = train_test_split(users, artists, playcounts, test_size=.2, random_state=42)

	for num_hidden_nodes in [1000]:
		for regularizer in [.001]:
			for batch_size in [128]:
				featureExtractor = AutoEncoderFeatureExtractor()
				featureExtractor.train_model_from_list(artists_train, users_train, playcounts_train, total_artists, total_users, num_hidden_nodes, regularizer, batch_size=batch_size)
				predicted_matrix = featureExtractor.predicted_ratings_matrix.T

				relevant_items = featureExtractor.get_matrix_from_list(users_test, artists_test, playcounts_test, total_users, total_artists)
				relevant_items = relevant_items > 0

				reqd_sorting = np.argsort(-predicted_matrix, axis=1)
				# print(reqd_sorting)
				# print(predicted_matrix[np.arange(relevant_items.shape[0]).reshape(relevant_items.shape[0], 1), reqd_sorting])
				# print(relevant_items[np.arange(relevant_items.shape[0]).reshape(relevant_items.shape[0], 1), reqd_sorting])

				relevant_items = relevant_items[np.arange(relevant_items.shape[0]).reshape(relevant_items.shape[0], 1), reqd_sorting]
				total_rated = np.sum(relevant_items, axis=1).reshape(relevant_items.shape[0], 1).astype(float)
				total_rated[total_rated == 0.] = 1.

				cummulative_relevence = np.cumsum(relevant_items, axis=1)

				recall = cummulative_relevence / total_rated
				precision = cummulative_relevence / np.cumsum(np.ones(relevant_items.shape), axis=1).astype(float)

				# print(recall)
				# print(precision)

				# top-n is top-24

				recall_at_reqd_n = recall[:, 23]
				
				prec_sum = 0.

				for uid in range(total_users):
					has_rated = total_rated[uid, 0]
					req_col = int(min(23, has_rated))
					
					prec_sum += precision[uid, req_col] if has_rated != 0 else 1

				error = 0.

				for user_id, artist_id, cnt in zip(users_test, artists_test, playcounts_test):
					error += np.abs(min(cnt, 300) - predicted_matrix[user_id][artist_id])
					# print cnt, predicted_matrix[user_id][artist_id]

				error /= users_test.size

				print(num_hidden_nodes, regularizer, batch_size, np.mean(recall_at_reqd_n), prec_sum / total_users, error / 300.)
