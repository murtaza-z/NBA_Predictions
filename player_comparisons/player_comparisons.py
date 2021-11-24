import pandas
import os
import tensorflow as tf
import numpy as np


file_dir = os.path.dirname(__file__)

columns = [
	'fg_per_g', 'fga_per_g', 'fg_pct',
	'fg3_per_g', 'fg3a_per_g', 'fg3_pct',
	'fg2_per_g', 'fg2a_per_g', 'fg2_pct',
	'efg_pct',
	'ft_per_g', 'fta_per_g', 'ft_pct',
	'orb_per_g', 'drb_per_g', 'trb_per_g',
	'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g'
]

def readTable(filename: str):
    table = pandas.read_csv(filename, index_col=0)
    return table

def constructDataset(dst: str):
	data_dir = "data"

	total_cols = ['season', 'points0', 'points1', 'team_stats0', 'team_stats1']
	
	all_game_data = []
	for year in range(2000, 2021):
		season_dir = "{dir}/{year}".format(dir=data_dir, year=year)
		game_results = readTable("{dir}/game_results.csv".format(dir=season_dir))
		game_results = game_results[["box_score_text", "visitor_team_name", "visitor_pts", "home_team_name", "home_pts"]]
		player_stats = readTable("{dir}/player_stats.csv".format(dir=season_dir))
		print(year)
		
		for (i, box_score_text, team0, team0_points, team1, team1_points) in game_results.itertuples():
			teams = [team0, team1]
			game_data = [year, team0_points, team1_points]
			for team in teams:
				game_team = "{game}_{team}".format(game=box_score_text, team=team)
				team_table = readTable("{dir}/{csv}.csv".format(dir=season_dir, csv=game_team))
				players = team_table["player"]

				players = pandas.concat([
					player_stats[
						(player_stats['player'].str.match(r'^' + player + r'\*?$')) & (player_stats['team_id'] == team)
					][columns]
					for player in players
				])

				game_data += [players.to_numpy()]
			all_game_data += [game_data]
	df = pandas.DataFrame(all_game_data, columns=total_cols)

	print(df)
	df.to_csv(dst)
	return df

def fixNumpyStrings(df):
	for team_stats in [df['team_stats0'], df['team_stats1']]:
		for i in range(len(team_stats)):
			players_str = team_stats.iloc[i].strip()
			players_str = players_str[1:-1]

			players_strs = players_str.replace(']', '').split('[')[1:]
			players_splits = [player_str.split() for player_str in players_strs]

			player_stats = np.empty((len(players_splits), len(players_splits[0])))
			for player in range(player_stats.shape[0]):
				for col in range(player_stats.shape[1]):
					value = players_splits[player][col]
					if value == 'nan':
						value = 0
					else:
						value = float(value)
					player_stats[player][col] = value
			team_stats.iloc[i] = player_stats
		print(team_stats)
	return df

def balanceDataset(df):
	# NOTE: the data could be mirror here so that there are equal wins and losses for home vs visitor wins;
	# however, this was avoided in case the home-field advantage is a real thing

	indices0 = np.where(df['points0'] > df['points1'])[0]
	indices1 = np.where(df['points0'] < df['points1'])[0]
	len0 = len(indices0)
	len1 = len(indices1)
	if len0 > len1:
		remove_amount = len0 - len1
		indices_to_remove = np.random.choice(indices0, size=remove_amount, replace=False)
	elif len0 < len1:
		remove_amount = len1 - len0
		indices_to_remove = np.random.choice(indices1, size=remove_amount, replace=False)
	else:
		return df

	return df.drop(indices_to_remove)

def standardizeDataset(df):
	import json
	seasons = df['season'].unique()
	season_min = seasons.min()
	season_max = seasons.max()
	standards_file = 'player_cmp_standards_{min}-{max}.json'.format(min=season_min, max=season_max)
	standards_file = os.path.join(file_dir, standards_file)

	if not os.path.exists(standards_file):
		data_dir = "data"
		mins = []
		maxs = []
		for year in seasons:
			season_dir = "{dir}/{year}".format(dir=data_dir, year=year)
			player_stats = readTable("{dir}/player_stats.csv".format(dir=season_dir))
			player_stats = player_stats[columns]
			mins += [player_stats.min()]
			maxs += [player_stats.max()]

		true_mins = mins[0]
		true_maxs = maxs[0]
		for year in range(1, len(mins)):
			for col in range(1, len(mins[year])):
				if (maxs[year][col] > true_maxs[col]):
					true_maxs[col] = maxs[year][col]
				if (mins[year][col] < true_mins[col]):
					true_mins[col] = mins[year][col]

		data = {
			'min': true_mins.to_numpy().tolist(),
			'max': true_maxs.to_numpy().tolist()
		}
		with open(standards_file, 'w') as jsonfile:
			json.dump(data, jsonfile)

	with open(standards_file, 'r') as jsonfile:
		data = json.load(jsonfile)
	maxs = np.asarray(data['max'])
	mins = np.asarray(data['min'])

	value_spread = maxs - mins
	
	all_team_stats = [df['team_stats0'], df['team_stats1']]
	for team_stats in all_team_stats:
		for players in team_stats:
			for player in players:
				new_player = (player - mins) / value_spread
				for i in range(len(new_player)):
					player[i] = new_player[i]

	return df

csv_filename = "players_comparisons.csv"
csv_filename = os.path.join(file_dir, csv_filename)
if not os.path.exists(csv_filename):
	constructDataset(csv_filename)

df = readTable(csv_filename)

df = fixNumpyStrings(df)
print(df)
df = balanceDataset(df)
df = standardizeDataset(df)
print(df)
x = df[df.columns.drop(['season', 'points0', 'points1'])]
print(x)
y = (df['points0'] < df['points1']).astype(int)
print(y)
split_ind = df.shape[0] * 6 // 10
train_x = x.iloc[:split_ind]
train_y = y.iloc[:split_ind]
test_x  = x.iloc[split_ind:]
test_y  = y.iloc[split_ind:]

def create_model(max_length: int, width: int):
	print(max_length)
	print(width)
	team1_input = tf.keras.Input(shape=(max_length, width,), name='team0')
	team2_input = tf.keras.Input(shape=(max_length, width,), name='team1')
	
	rnn = tf.keras.layers.GRU(128)
	team1_rnn = rnn(team1_input)
	team2_rnn = rnn(team2_input)
	
	concat_layer = tf.keras.layers.Concatenate()([team1_rnn, team2_rnn])
	dense_layer = tf.keras.layers.Dense(units=128, activation='relu')(concat_layer)
	dense_layer = tf.keras.layers.Dense(units=64, activation='relu')(dense_layer)
	dense_layer = tf.keras.layers.Dense(units=32, activation='relu')(dense_layer)
	pred_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense_layer)
	
	model = tf.keras.Model(
		inputs=[team1_input, team2_input],
		outputs=[pred_layer]
	)


	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
		loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
		metrics=['binary_accuracy']
	)
	return model

def train_model(train_features, train_labels, model, epochs=30, batch_size=100, validation_split=0.1):

	history = model.fit(x=train_features, y=train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
	return history


input0 = tf.ragged.constant([nparray.tolist() for nparray in train_x['team_stats0']]).to_tensor()
input1 = tf.ragged.constant([nparray.tolist() for nparray in train_x['team_stats1']]).to_tensor()

model = create_model(int(input0.shape[1]), int(input0.shape[2]))

saved_weights_file = 'saved_model_player_cmp'
saved_weights_file = os.path.join(file_dir, saved_weights_file)
model.load_weights(filepath=saved_weights_file)
# history = train_model({'team0': input0, 'team1': input1}, train_y, model, 100, 1000)
# model.save_weights(filepath=saved_weights_file)

input0 = tf.ragged.constant([nparray.tolist() for nparray in test_x['team_stats0']]).to_tensor()
input1 = tf.ragged.constant([nparray.tolist() for nparray in test_x['team_stats1']]).to_tensor()
test_loss, test_acc = model.evaluate(x={'team0': input0, 'team1': input1}, y=test_y)
print(test_loss, test_acc)

predictions = model.predict(x={'team0': input0, 'team1': input1})
print(predictions)
sum = 0
for prediction in predictions:
    sum += predictions[0]
mean_prediction = sum / len(predictions)
print(mean_prediction)
