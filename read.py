import pandas

def readTable(filename: str):
    table = pandas.read_csv(filename, index_col=0)
    return table

table = readTable("data/2020/game_results.csv")
print(table)
print(table.columns)
print(table['visitor_team_name'])