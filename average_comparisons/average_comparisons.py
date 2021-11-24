import pandas
import os
import tensorflow as tf
import numpy as np


file_dir = os.path.dirname(__file__)

def readTable(filename: str):
    table = pandas.read_csv(filename, index_col=0)
    return table

def constructDataset(dst: str):
	data_dir = "data"
	columns = [
		'fg_per_g', 'fga_per_g', 'fg_pct',
		'fg3_per_g', 'fg3a_per_g', 'fg3_pct',
		'fg2_per_g', 'fg2a_per_g', 'fg2_pct',
		'efg_pct',
		'ft_per_g', 'fta_per_g', 'ft_pct',
		'orb_per_g', 'drb_per_g', 'trb_per_g',
		'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g'
	]

	team_cols = [
		[col + str(i) for col in columns] for i in range(2)
	]
	total_cols = ['season', 'points0', 'points1'] + team_cols[0] + team_cols[1]
	
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

				game_data += list(players.mean())
			all_game_data += [game_data]
	df = pandas.DataFrame(all_game_data, columns=total_cols)

	print(df)
	df.to_csv(dst)
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
    return (df - df.min()) / (df.max() - df.min())
    # return (df - df.mean()) / df.std()

csv_filename = "average_comparisons.csv"
csv_filename = os.path.join(file_dir, csv_filename)
if not os.path.exists(csv_filename):
	constructDataset(csv_filename)

df = readTable(csv_filename)

df = balanceDataset(df)
x = df[df.columns.drop(['season', 'points0', 'points1'])]
x = standardizeDataset(x)
print(x)
y = (df['points0'] < df['points1']).astype(int)
split_ind = df.shape[0] * 6 // 10
train_x = x.iloc[:split_ind]
train_y = y.iloc[:split_ind]
test_x  = x.iloc[split_ind:]
test_y  = y.iloc[split_ind:]

def create_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(units=128, activation='relu'),
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=1, activation='sigmoid')
	])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
		loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
		metrics=['binary_accuracy']
	)
	return model

def train_model(train_features, train_labels, model, epochs=30, batch_size=100, validation_split=0.1):

	history = model.fit(x=train_features, y=train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
	return history

model = create_model()

saved_weights_file = 'saved_model_average_cmp'
saved_weights_file = os.path.join(file_dir, saved_weights_file)
model.load_weights(filepath=saved_weights_file)
# history = train_model(train_x, train_y, model, 3000, 10000)
# model.save_weights(filepath=saved_weights_file)

test_loss, test_acc = model.evaluate(x=test_x, y=test_y)
print(test_loss, test_acc)

predictions = model.predict(x=test_x)
print(predictions)
sum = 0
for prediction in predictions:
    sum += predictions[0]
mean_prediction = sum / len(predictions)
print(mean_prediction)
