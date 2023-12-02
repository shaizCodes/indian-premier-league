import pandas
import ast
import matplotlib.pyplot as plotter
import numpy

MATCHES = pandas.read_csv(".\\datasets\\IPL_Matches_2008_2022.csv")
BALLS = pandas.read_csv(".\\datasets\\IPL_Ball_by_Ball_2008_2022.csv")

SEASONS = {
    1: '2007/08',   2: '2009',  3: '2009/10',   4: '2011',  5: '2012',
    6: '2013',  7: '2014',  8: '2015',  9: '2016',  10: '2017',
    11: '2018', 12: '2019', 13: '2020/21',  14: '2021', 15: '2022'
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


def printHeader():
    HEADER = "INDIAN PREMIER LEAGUE DATA"
    HEADER_WIDTH = len(HEADER)+10
    print(f"{'='*HEADER_WIDTH}\n||{HEADER.center(HEADER_WIDTH-4)}||\n{'='*HEADER_WIDTH}")

def chooseSeason():
    while True:
        global SEASONS
        menu = "-"*40+"\nAVAILABLE SEASONS\n"
        count = 0
        for key, season in SEASONS.items():
            menu += ("("+str(key)+"). "+season).ljust(20)
            count += 1
            if count%3==0: menu += "\n"
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
        menu = "-"*40+"\nAVAILABLE TEAMS\n"
        if season=='all':
            global TEAMS
            count = 0
            for key, team in TEAMS.items():
                menu += ("("+str(key)+"). "+(team if key<10 else team[0])).ljust(35)
                count += 1
                if count%3 == 0: menu += '\n'
            menu += "(0). Select all Teams\n"
            menu += "Choose Team[0 - "+str(len(TEAMS))+"]: "
            choice = int(input(menu))
            if choice > -1 and choice < 14:
                if choice == 0:
                    return 'all'
                return TEAMS[choice]
        else:
            global MATCHES
            seasons = MATCHES[MATCHES['Season']==season]
            teams = seasons['Team1'].unique()
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
                return teams[choice-1]
        print(f"Invalid Input! Please select a valid option.")

def choosePlayer(team='all', season='all'):
    global MATCHES
    while True:
        players = []
        if season=='all':
            if team=='all':
                squads = MATCHES['Team1Players'].apply(ast.literal_eval)
                players = getPlayers(squads)
            else:
                squads = MATCHES[(MATCHES['Team1']==team)]['Team1Players'].apply(ast.literal_eval) 
                players = getPlayers(squads)
                squads = MATCHES[(MATCHES['Team2']==team)]['Team2Players'].apply(ast.literal_eval)
                players = getPlayers(squads)
        else:
            if team=='all':
                matches = MATCHES[MATCHES['Season']==season]
                squads = matches['Team1Players'].apply(ast.literal_eval)
                players = getPlayers(squads)
                squads = matches['Team2Players'].apply(ast.literal_eval)
                players = getPlayers(squads)
            else:
                squads = MATCHES[(MATCHES['Team1']==team) & (MATCHES['Season']==season)]['Team1Players'].apply(ast.literal_eval)
                players = getPlayers(squads)
                squads = MATCHES[(MATCHES['Team2']==team) & (MATCHES['Season']==season)]['Team2Players'].apply(ast.literal_eval)
                players = getPlayers(squads)
        menu = "-"*40+"\nAVAILABLE PLAYERS\n"
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

def getPlayers(squads):
    players = []
    for squad in squads:
        for player in squad:
            if not player in players:
                players.append(player)
    return players

def plotPlayerPerformance(player='all', team='all', season='all'):
    matches = MATCHES
    balls = BALLS
    runsPerBall = []
    runsPerInning = []
    runsPerSeason = []
    comulativeRuns = []
    if not season=='all':
        matches = matches[matches['Season']==season]
    if not team=='all':
        matches = matches[(matches['Team1']==team) | (matches['Team2']==team)]
    if not player=='all':
        balls = pandas.merge(matches, balls[balls['batter']==player], on='ID')
    balls = pandas.merge(matches, balls, on='ID')
    runsPerBall = balls.groupby(['ID', 'innings', 'overs', 'ballnumber'], as_index=True)['total_run'].aggregate('sum').to_list()
    runsPerInning = balls.groupby(['ID', 'innings'], as_index=True)['total_run'].aggregate("sum").to_list()
    runsPerSeason = balls.groupby(['Season'], as_index=True)['total_run'].aggregate('sum').to_list()
    comulativeRuns = balls['total_run'].cumsum().to_list()
    figure, axes = plotter.subplots(nrows=2, ncols=2)
    plotter.suptitle(player+"\'s Batting Performance")
    
    axes[0, 0].set_title("Runs per Ball")
    axes[0, 0].plot(runsPerBall)
    X = numpy.array([n for n in range(len(runsPerBall))])
    y = numpy.array(runsPerBall)
    gradient = sum((X-X.mean())*(y-y.mean()))/sum((X-X.mean())**2)
    y_intercept = y.mean()-gradient*X.mean()
    
    def linear_model(X): return gradient*X+y_intercept
    
    axes[0, 0].plot(X, linear_model(X), 'r-')

    axes[0, 1].set_title("Runs per Inning")
    axes[0, 1].plot(runsPerInning)
    X = numpy.array([n for n in range(len(runsPerInning))])
    y = numpy.array(runsPerInning)
    gradient = sum((X-X.mean())*(y-y.mean()))/sum((X-X.mean())**2)
    y_intercept = y.mean()-gradient*X.mean()
    axes[0, 1].plot(X, linear_model(X), 'r-')

    axes[1, 0].set_title("Runs per Season")
    axes[1, 0].plot(runsPerSeason)
    X = numpy.array([n for n in range(len(runsPerSeason))])
    y = numpy.array(runsPerSeason)
    gradient = sum((X-X.mean())*(y-y.mean()))/sum((X-X.mean())**2)
    y_intercept = y.mean()-gradient*X.mean()
    axes[1, 0].plot(X, linear_model(X), 'r-')

    axes[1, 1].set_title("Commulative Runs in Balls")
    axes[1, 1].plot(comulativeRuns, label='Actual Data')
    X = numpy.array([n for n in range(len(comulativeRuns))])
    y = numpy.array(comulativeRuns)
    gradient = sum((X-X.mean())*(y-y.mean()))/sum((X-X.mean())**2)
    y_intercept = y.mean()-gradient*X.mean()
    axes[1, 1].plot(X, linear_model(X), 'r-', label='Prediction')
    plotter.legend()
    plotter.tight_layout()
    plotter.show() 

try:
    printHeader()
    season = chooseSeason()
    team = chooseTeam(season)
    if isinstance(team, str):
        player = choosePlayer(team, season)
        print(f"Selected Player: {player} (Team: {team}) (Season: {season})")
        plotPlayerPerformance(player, team, season)
    else: # which means a team has multiple names
        print("To be continued...")
except Exception as error:
    print(f"An error occured...ERROR: {error}")
