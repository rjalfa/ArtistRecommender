import numpy as np

def get_data_as_list(file_name):
	num_users, num_artists = 0, 0
	user_index_mapping, artist_index_mapping = {}, {}
	
	with open(file_name) as data_file:
		for line in data_file:
			user_id, artist_id, playcount = line.split()
			
			if user_id not in user_index_mapping:
				user_index_mapping[user_id] = num_users
				num_users += 1

			if artist_id not in artist_index_mapping:
				artist_index_mapping[artist_id] = num_artists
				num_artists += 1

	print('num_users:', num_users, 'num_artists:', num_artists)

	users, artists, playcounts = [], [], []
	with open(file_name) as data_file:
		for line in data_file:
			user_id, artist_id, playcount = line.split()
			
			users.append(user_index_mapping[user_id])
			artists.append(artist_index_mapping[artist_id])
			playcounts.append(int(playcount))

	users, artists, playcounts = np.array(users), np.array(artists), np.array(playcounts)

	return users, artists, playcounts, num_users, num_artists

def get_data_as_matrix(file_name):
	num_users, num_artists = 0, 0
	user_index_mapping, artist_index_mapping = {}, {}
	
	with open(file_name) as data_file:
		for line in data_file:
			user_id, artist_id, playcount = line.split()
			
			if user_id not in user_index_mapping:
				user_index_mapping[user_id] = num_users
				num_users += 1

			if artist_id not in artist_index_mapping:
				artist_index_mapping[artist_id] = num_artists
				num_artists += 1

	print('num_users:', num_users, 'num_artists:', num_artists)

	playcount_matrix = [[0 for i in range(num_artists)] for j in range(num_users)]

	with open(file_name) as data_file:
		for line in data_file:
			user_id, artist_id, playcount = line.split()
			playcount_matrix[user_index_mapping[user_id]][artist_index_mapping[artist_id]] = int(playcount)

	playcount_matrix = np.array(playcount_matrix)

	return playcount_matrix, num_users, num_artists

if __name__ == "__main__":
	pc_matrix, uidx_map, aidx_map = get_data_as_matrix('filtered_data.tsv')
	users, artists, playcounts, uidx_map, aidx_map = get_data_as_list('filtered_data.tsv')
	# print pc_matrix
	# print users
	# print artists
	# print playcounts