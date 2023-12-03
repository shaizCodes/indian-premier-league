import pandas
import ast
import matplotlib.pyplot as plotter
import numpy

# MACHINE LEARNING MODEL
gradient = 0
y_intercept = 0
def univariate_linear_model(X):
    return gradient * X + y_intercept

MATCHES = pandas.read_csv(".\\datasets\\IPL_Matches_2008_2022.csv")
BALLS = pandas.read_csv(".\\datasets\\IPL_Ball_by_Ball_2008_2022.csv")

SEASONS = {
    1: '2007/08',   2: '2009',    3: '2009/10',    4: '2011',  5: '2012',
    6: '2013',      7: '2014',    8: '2015',       9: '2016',  10: '2017',
    11: '2018',    12: '2019',   13: '2020/21',   14: '2021',  15: '2022'
}
TEAMS = {
    1: 'Kolkata Knight Riders',
    2: 'Chennai Super Kings',
    3: 'Mumbai Indians',
    4: 'Royal Challengers Bangalore',
    5: 'Rajasthan Royals',
    6: 'Deccan Chargers',
    7: 'Sunrisers Hyderabad',
    8: 'Lucknow Super Giants',
    9: 'Kochi Tuskers Kerala',
    10: ['Punjab Kings', 'Kings XI Punjab'],
    11: ['Delhi Capitals', 'Delhi Daredevils'],
    12: ['Gujarat Titans', 'Gujarat Lions'],
    12: ['Rising Pune Supergiant', 'Rising Pune Supergiants', 'Pune Warriors']
}

# For the sake of formatting Console MENU
WIDTH = 25

def printHeader(header):
    border = "="*(WIDTH*4)
    print(f"{border.center(WIDTH*5)}\n{('|| '+(header.center(WIDTH*4-6))+' ||').center(WIDTH*5)}\n{border.center(WIDTH*5)}")

def chooseSeason():
    while True:
        menu = "-"*(WIDTH*5)+"\n"+("AVAILABLE SEASONS").center(WIDTH*5)+"\n"
        count = 0
        for key, season in SEASONS.items():
            menu += ("("+str(key)+"). "+season).ljust(WIDTH)
            count += 1
            if count%5==0: menu += "\n"
        menu += "(0). Select all Seasons\n"
        menu += "Choose Season[0 - "+str(count)+"]: "
        choice = int(input(menu))
        if choice > -1 and choice < 16:
            if choice == 0:
                return 'all'
            return SEASONS[choice]
        print(f"Invalid Input! Please select a valid option.")

def chooseTeam(season='all'):
    while True:
        menu = "-"*(WIDTH*5)+"\n"+("AVAILABLE TEAMS").center(WIDTH*5)+"\n"
        teams = []
        if not season=='all':
            seasons = MATCHES[MATCHES['Season']==season]
            teams = seasons['Team1'].unique()
        else:
            for key, team in TEAMS.items():
                if not team in teams:
                    teams.append(team if isinstance(team, str) else team[0])
        count = 0
        for team in teams:
            count += 1
            menu += "({}). {}".format(count, team).ljust(35)
            if count%3==0: menu += '\n'
        menu += "(0). Select all Teams\n"
        menu += "Choose Team[0 - "+str(count)+"]: "
        choice = int(input(menu))
        if choice > -1 and choice <= count:
            if choice == 0:
                return 'all'
            return TEAMS[choice] if season=='all' else teams[choice-1]
        print(f"Invalid Input! Please select a valid option.")

def choosePlayer(team='all', season='all'):
    while True:
        players = []
        squads = []
        matches = MATCHES
        if not season=='all':
            matches = matches[matches['Season']==season]
        if isinstance(team, str):
            if not team=='all':
                squads = matches[(matches['Team1']==team)]['Team1Players'].apply(ast.literal_eval)
                updatePlayers(players, squads)
                squads = matches[(matches['Team2']==team)]['Team2Players'].apply(ast.literal_eval)
                updatePlayers(players, squads)
            else:
                squads = matches['Team1Players'].apply(ast.literal_eval)
                updatePlayers(players, squads)
                squads = matches['Team2Players'].apply(ast.literal_eval)
                updatePlayers(players, squads)
        else:
            players = getPlayers(players, team, matches)
        menu = "-"*(WIDTH*5)+"\n"+("AVAILABLE PLAYERS").center(WIDTH*5)+"\n"
        count = 0
        for player in players:
            count += 1
            menu += "({}). {}".format(str(count), player).ljust(30)
            if count%5 ==0: menu += '\n'
        menu += "\n(0). Select all players\n"
        menu += "Select Player[0 - "+str(len(players))+"]: "
        choice = int(input(menu))
        if choice > -1 and choice <= len(players):
            if choice == 0:
                return 'all'
            return players[choice-1]
        print(f"Invalid Input! Please select a valid option.")

def updatePlayers(players, squads):
    for squad in squads:
        for player in squad:
            if not player in players:
                players.append(player)

def getPlayers(players, teams, matches):
    players = []
    times = 0
    if not isinstance(team, str):
        times = len(team)
    time = 0
    while time<times:
        squads = matches[matches['Team1']==teams[time]]['Team1Players'].apply(ast.literal_eval)
        updatePlayers(players, squads)
        squads = matches[matches['Team2']==teams[time]]['Team1Players'].apply(ast.literal_eval)
        updatePlayers(players, squads)
        time += 1
    return players

def plotPlayerPerformance(player='all', team='all', season='all'):
    matches = MATCHES
    if not season=='all':
        matches = matches[matches['Season']==season]
    if not team=='all':
        if not isinstance(team, str):
            nTeams = len(team)
            dataFrames = []
            index = 0
            while index < nTeams:
                dataFrames.append(matches[(matches['Team1']==team[index]) | (matches['Team2']==team[index])])
                index += 1
            matches = pandas.concat(dataFrames, ignore_index=True)
        else:
            matches = matches[(matches['Team1']==team) | (matches['Team2']==team)]
    balls = pandas.merge(matches, BALLS, on='ID')
    if not player=='all':
        balls = pandas.merge(matches, BALLS[BALLS['batter']==player], on='ID')
    runs = {}
    runs['PER_BALL'] = balls.groupby(['ID', 'innings', 'overs', 'ballnumber'], as_index=True)['total_run'].aggregate('sum').to_list()
    runs['PER_INNING'] = balls.groupby(['ID', 'innings'], as_index=True)['total_run'].aggregate("sum").to_list()
    runs['PER_SEASON'] = balls.groupby('Season', as_index=True)['total_run'].aggregate('sum').to_list()
    runs['COMULATIVE'] = balls['total_run'].cumsum().to_list()
    labels = {
        'PER_BALL': 'Runs per Ball',
        'PER_INNING': 'Runs per Inning',
        'PER_SEASON': 'Runs per Season',
        'COMULATIVE': 'Comulative Runs in Balls'
    }
    figure, axes = plotter.subplots(nrows=2, ncols=2)
    plotter.suptitle(player+"\'s Batting Performance")
    global gradient, y_intercept
    indices, i = [[0, 0], [0, 1], [1, 0], [1, 1]], 0
    for category, score in runs.items():
        axes[indices[i][0], indices[i][1]].set_title(labels[category])
        if len(score)>1:
            X = numpy.array([n+1 for n in range(len(score))])
            y = numpy.array(score)
            gradient = sum((X-X.mean())*(y-y.mean()))/sum((X-X.mean())**2)
            y_intercept = y.mean()-gradient*X.mean()
            axes[indices[i][0], indices[i][1]].plot(X, score)
            axes[indices[i][0], indices[i][1]].plot(X, univariate_linear_model(X), 'r--')
        else:
            axes[indices[i][0], indices[i][1]].scatter([n+1 for n in range(len(score))], score)
        i += 1
    plotter.tight_layout()
    plotter.show() 

try:
    printHeader("INDIAN PREMIER LEAGUE DATA")
    season = chooseSeason()
    team = chooseTeam(season)
    player = choosePlayer(team, season)
    print(f"Selected Player: {player} (Team: {team}) (Season: {season})")
    plotPlayerPerformance(player, team, season)
except Exception as error:
    print(f"An error occured...ERROR: {error}")
