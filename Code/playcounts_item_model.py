import sys
import pandas as pd
import numpy as np
import numpy.random as rnd
from scipy import sparse
from sklearn.metrics import pairwise
from sklearn.model_selection import KFold, train_test_split
import get_data

class MemoryBasedModel:
	def get_matrix_from_list(self, rows, columns, ratings, total_rows, total_columns):
		ratings_matrix = [[0 for i in range(total_columns)] for j in range(total_rows)]

		for row_id, column_id, val in zip(rows, columns, ratings):
			ratings_matrix[row_id][column_id] = val

		return np.array(ratings_matrix)

	def train_model_from_list(self, rows, columns, ratings, total_rows, total_columns, alpha=.5, significance_scale=1):
		ratings_matrix = self.get_matrix_from_list(rows, columns, ratings, total_rows, total_columns)

		self.train_model_from_matrix(ratings_matrix, alpha, significance_scale)

	def train_model_from_matrix(self, ratings_matrix, alpha=.5, significance_scale=1):
		self.ratings_matrix = ratings_matrix
		self.has_rated_matrix = ratings_matrix > 0

		self.similarity_matrix = self.calculate_similarities(self.has_rated_matrix, alpha, significance_scale)

	def calculate_similarities(self, ratings_matrix, alpha=.5, significance_scale=1):
		idf_scores = np.sum(ratings_matrix, axis=0)
		idf_scores[idf_scores == 0] = 1
		idf_scores = np.nan_to_num(np.log(ratings_matrix.shape[0] / idf_scores))

		magnitudes = np.sum(ratings_matrix, axis=1)
		first_term_scaled = magnitudes ** alpha
		second_term_scaled = magnitudes ** (1 - alpha)

		denominators = np.outer(first_term_scaled, second_term_scaled) ** significance_scale
		denominators[denominators == 0] = 1

		neighbours = np.dot(ratings_matrix * idf_scores, np.transpose(ratings_matrix)) / denominators

		return neighbours;

	def impute_missing_values(self, sim_threshold=0.):
		predicted_matrix = self.similarity_matrix.dot(self.has_rated_matrix)

		return predicted_matrix


	# returns the 'k' best neighbours of user 'uid' having similarity above sim_threshold and have rated item 'iid'
	def get_k_best_neighbours(self, uid, iid, k = -1, sim_threshold = -1):
		indices = np.where((self.user_item_matrix[:, iid] != 0) & (np.absolute(self.similarity_matrix[uid]) >= sim_threshold))
		indices = indices[0][np.argsort(self.similarity_matrix[uid][indices])]
		
		if (k == -1 or indices.size <= k):
			return indices
		else:
			return indices[-k:]

	# predicts the rating given by user 'uid' to item 'iid'
	def predict(self, rid, cid, k_best = -1, sim_threshold = -1):
		return self.predicted_matrix[rid][cid]

	def test_model_from_file(self, file_name, k_best = -1, sim_threshold = -1):
		users, items, ratings = self.get_user_item_list(file_name)
		return self.test_model(users, items, ratings, k_best, sim_threshold)

	def test_model(self, users, items, ratings, k_best = -1, sim_threshold = -1):
		vpredict = np.vectorize(lambda uid, iid: self.predict(uid, iid, k_best, sim_threshold))
		predictions = vpredict(users, items)
		
		# make the predicts ratings obey the scale
		predictions[np.where(predictions > 5)] = 5
		predictions[np.where(predictions < 1)] = 1

		return np.mean(np.absolute(np.subtract(predictions, np.array(ratings))))

if __name__ == "__main__":
	users, artists, playcounts, total_users, total_artists = get_data.get_data_as_list('filtered_data.tsv')
	users_train, users_test, artists_train, artists_test, playcounts_train, playcounts_test = train_test_split(users, artists, playcounts, test_size=.2, random_state=42)

	memory_based_model = MemoryBasedModel()
	relevant_items = memory_based_model.get_matrix_from_list(users_test, artists_test, playcounts_test, total_users, total_artists)
	relevant_items = relevant_items > 0
	total_rated = np.sum(relevant_items, axis=1).reshape(relevant_items.shape[0], 1).astype(float)
	total_rated[total_rated == 0.] = 1.

	for q in [1, 2, 3]:
		for alpha in [.1, .3, .5, .7, .9]:
			rnd.seed(42)
			
			memory_based_model.train_model_from_list(artists_train, users_train, playcounts_train, total_artists, total_users, alpha, q)		

			predicted_matrix = memory_based_model.impute_missing_values().T
			predicted_matrix = predicted_matrix * (1 - memory_based_model.has_rated_matrix.T)

			reqd_sorting = np.argsort(-predicted_matrix, axis=1)

			ordered_relevant_items = relevant_items[np.arange(relevant_items.shape[0]).reshape(relevant_items.shape[0], 1), reqd_sorting]
			
			cummulative_relevence = np.cumsum(ordered_relevant_items, axis=1)

			recall = cummulative_relevence / total_rated
			precision = cummulative_relevence / np.cumsum(np.ones(ordered_relevant_items.shape), axis=1).astype(float)


			# top-n is top-24
			recall_at_reqd_n, prec_sum = recall[:, 9], 0.

			for uid in range(total_users):
				has_rated = total_rated[uid, 0]
				req_col = int(min(9, has_rated))
				
				prec_sum += precision[uid, req_col] if has_rated != 0 else 1

			print alpha, q, np.mean(recall_at_reqd_n), prec_sum / total_users
