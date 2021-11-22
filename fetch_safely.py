import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas
import os

def PullLogWrite(msg):
	"""A function to append a NBA download to the 'completed' log.

	Args:
		msg (str): Message id to represent which download was completed.
	"""
	with open("year_complete.txt", "a") as f:
		f.write(msg+'\n')

def PullLogRead():
	"""A function to retrieve the set of 'completed' NBA downloads.

	Returns:
		set(str): A set with which downloads have already been completed.
	"""
	if os.path.exists("year_complete.txt"):
		with open("year_complete.txt", "r") as f:
			return set(f.read().strip().split('\n'))
	return set()

def createRequest(url):
	s = requests.Session()
	s.auth = ('user', 'pass')
	page = s.get(url)
	if (not page.ok):
		error_msg = "ERROR - {0}: {1}".format(page.status_code, page.reason)
		print(error_msg)
		page.raise_for_status()
		return False
	
	return page

def getSoup(url):
	page = createRequest(url)
	if not page:
		return page
	soup = BeautifulSoup(page.text, 'html.parser')
	return soup

def retrieveMonthsInYear(year):
	"""
 	A function to receive the months for an NBA season.

	Args:
		year (int|str): An NBA season to receive its months for.

	Returns:
		str[]: The months within the given NBA season.
	"""
	url = "https://www.basketball-reference.com/leagues/NBA_{year}_games.html".format(year=year)
	month_divs = getSoup(url).find(id='content').find(class_='filter').find_all('div')
	months = [month.text.strip().replace(' ', '-').lower() for month in month_divs]
	return months

def retrieveTable(soup_html: str, table_id, append_cols: list = [], SelectColValueFunc = lambda col: (col.text, [])):
	"""
	A function to receive a specific NBA table from the specific url.
 
	Args:
		soup_html (str): The url to request its immediate html from.
		table_id (str|int): The DOM id to search for which corresponds to the requested table.
		append_cols (list, optional): Data columns to construct and append to the table. Defaults to [].
		SelectColValueFunc (function, optional): This function is used to edit a column value and
  			construct new appendable data using the same column. This is important to do here before all context
     		contained inside that column's DOM is loss asside from its text. Defaults to lambda col:(col.text, []).

	Returns:
		list[list]: A table with the first rows as headers and all proceeding rows as data from the requested table.
	"""
	table_html = soup_html.find(id=table_id)
	columns = table_html.thead.find_all('tr')[-1].find_all('th')
	col_names = [col['data-stat'] for col in columns] + append_cols

	table = [col_names]
	for row in table_html.tbody.find_all('tr'):
		# Ensure that this row of the table isn't a repeated header
		if not 'class' in row.attrs or row['class'][0] != 'thead':
			columns = row.children
			row_entry = []
			appendable_data = []
			for col in columns:
				# Ensure that this column has data
				# 'data-stat' stores the names of the column is the table headers it corresponds to
				if col['data-stat']:
					# Have an opportunity to edit this column and also construct new data to be appended to the row
					value, add_cols = SelectColValueFunc(col)
					row_entry += [value]
					appendable_data += add_cols
			# Ensure that the appendable_data is append to the end of this row
			row_entry += appendable_data
			table += [row_entry]
	return table

def retrievePlayerStats(year, stat_type):
	"""
	A function to request player statistics for a single NBA season.

	Args:
		year (str|int): The NBA season to request player statistics for.
		stat_type (str): The stat type to request (e.g. "totals", "per_game", "per_minute", ect.)

	Returns:
		list[list]: A table for the provided NBA season with each players targeted statistics.
	"""
	url = "https://www.basketball-reference.com/leagues/NBA_{year}_{stat_type}.html".format(year=year, stat_type=stat_type)
	table_id = "{stat_type}_stats".format(stat_type=stat_type)

	def SelectColValueFunc(col):
		value = col.text
		add_cols = []
		if col['data-stat'] == 'player':
			player_id = "/".join(col.a['href'].split('/')[-1:]).split('.')[0]
			add_cols += [player_id]
		return value, add_cols

	return retrieveTable(getSoup(url), table_id, ['player_id'], SelectColValueFunc)

def retrievePlayerGamesInYear(player_id, year):
	"""
	A function to request each game's statistics for a single player in a given NBA season.

	Args:
		player_id (str): NBA player to requests game stats for.
		year (str|int): NBA season to request for.

	Returns:
		list[list]: A table where each row holds a different game's statistics for the given player in the given season.
	"""
	url = "https://www.basketball-reference.com/players/{player_id}/gamelog/{year}".format(player_id=player_id, year=year)
	table_id = 'pgl_basic'

	gameLog = retrieveTable(getSoup(url), table_id)
	gamesPerTeamDict = {}
	col_team = gameLog[0].index('team_id')
	gamesPerTeamDict["cols"] = gameLog[0]
	for game in gameLog[1:]:
		# Check if stats are missing
		print(str(len(game)) + " " + str(len(gameLog[0])))
		if len(game) >= len(gameLog[0]):
			team = game[col_team]
			if team not in gamesPerTeamDict:
				gamesPerTeamDict[team] = []
			gamesPerTeamDict[team] += [game]
	return gamesPerTeamDict

def retrieveGamesInYearMonth(year, month):
	"""
	A function to request the game results for a month in an NBA season.

	Args:
		year (str|int): NBA season to request games for.
		month (str): Specific month in the NBA season to request games for.

	Returns:
		list[list]: A table with each row holds the results for a specific game in the given month of the given NBA season.
	"""
	url = "https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html".format(year=year, month=month)
	table_id = 'schedule'

	def SelectColValueFunc(col):
		value = col.text
		add_cols = []
		if col['data-stat'] == 'date_game':
			value = str(datetime.strptime(col.text, r'%a, %b %d, %Y').date())
		elif col['data-stat'] in ['visitor_team_name', 'home_team_name']:
			value = col.a['href'].split('/')[2]
		elif col['data-stat'] == 'box_score_text':
			value = col.a['href'].split('/')[-1].split('.')[0]
		return value, add_cols

	return retrieveTable(getSoup(url), table_id, [], SelectColValueFunc)

def updateGameResults(year):
	"""
 	A function to retrieve all game results for the given NBA season.
	Also updates log that the season game results have been downloaded.

	Args:
		year (str|int): NBA season to request game results for.
	"""
	months = retrieveMonthsInYear(year)
	gamesInYear = None
	for month in months:
		gamesInYearMonth = retrieveGamesInYearMonth(year, month)
		if gamesInYear is None:
			gamesInYear = [gamesInYearMonth[0]]
		gamesInYear += gamesInYearMonth[1:]

	# Save to file
	dataFrame = pandas.DataFrame(gamesInYear[1:], columns=gamesInYear[0])
	dataFrame.to_csv("game_results.csv")
	PullLogWrite("game_results")

def updatePlayerStats(year):
	"""
	A function to retrieve all season-average player statistics for the given NBA season.
	Also updates log that the season player statistics have been downloaded.

	Args:
		year (str|int): NBA season to request player statistics for.
	"""
	table = retrievePlayerStats(year, 'per_game')

	# Save to file
	dataFrame = pandas.DataFrame(table[1:], columns=table[0])
	dataFrame.to_csv("player_stats.csv")
	PullLogWrite("player_stats")


def retrieveGameStatsForTeam(boxscore_url_id: str, teams):
	"""
	A function to retrieve all game-specific player statistics for a given NBA game.

	Args:
		boxscore_url_id (str): A url id for the game results.
		teams (list[str]): The abbreviations for the two teams which played in the game.

	Returns:
		list[list[list]]: A list of the two tables where each table corresponds to one team's player statistics for the game.
	"""
	url = "https://www.basketball-reference.com/boxscores/{game}.html".format(game=boxscore_url_id)
	soup_html = getSoup(url)

	tables = {}
	for team in teams:
		# table_id = "box-{team}-game-advanced".format(team=team)
		table_id = "box-{team}-game-basic".format(team=team)
		table = retrieveTable(soup_html, table_id)

		# Remove players who didn't participate
		full_row_len = len(table[0])
		table = [row for row in table if len(row) == full_row_len]
		tables[team] = table

	return tables

def updateGames():
	"""
	A function which completes the downloads for all game-specific player statistics.
	Also updates log for which game-specific statistics have been downloaded.
	"""
	complete_log = PullLogRead()
	game_results = pandas.read_csv("game_results.csv", index_col=0)
	game_results = game_results[["box_score_text", "visitor_team_name", "home_team_name"]]
	for (i, box_score_text, team1, team2) in game_results.itertuples():
		if box_score_text not in complete_log:
			print("updating game {0}...".format(box_score_text))
			teams = [team1, team2]
			tables = retrieveGameStatsForTeam(box_score_text, teams)
			for team in tables:
				game_team = "{game}_{team}".format(game=box_score_text, team=team)
				table = tables[team]

				# Save to file
				dataFrame = pandas.DataFrame(table[1:], columns=table[0])
				dataFrame.to_csv("{game_team}.csv".format(game_team=game_team))
			PullLogWrite(box_score_text)


def pullDataForYear(year):
	old_cwd = os.getcwd()
	cur_cwd = os.path.join(old_cwd, str(year))
	if not os.path.exists(cur_cwd):
		os.makedirs(cur_cwd)
	os.chdir(cur_cwd)

	complete_log = PullLogRead()
	if "game_results" not in complete_log:
		print("updating game_results...")
		updateGameResults(year)
	
	if "player_stats" not in complete_log:
		print("updating player_stats...")
		updatePlayerStats(year)
	
	updateGames()

	os.chdir(old_cwd)

def pullAll():
	old_cwd = os.getcwd()
	data_dir = "data"
	if not os.path.exists(data_dir):
		zip_dir = data_dir + ".zip"
		if os.path.exists(zip_dir):
			import shutil
			print("unzipping NBA data...")
			shutil.unpack_archive(zip_dir)
		else:
			os.makedirs(data_dir)
	os.chdir(data_dir)
	for year in range(2020, 1981, -1):
		pullDataForYear(year)
	os.chdir(old_cwd)

pullAll()