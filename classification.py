import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

label_encoder = LabelEncoder()
models = [DecisionTreeClassifier(), RandomForestClassifier(n_estimators=5, random_state=1)]
model_names = ['Decision Tree', 'Random Forest']

matches = pd.read_csv(".\\datasets\\IPL_Matches_2008_2022.csv")
matches['Result'] = matches['WinningTeam']==matches['Team1']
columns = [
    'ID', 'City', 'Date', 'Season', 'MatchNumber', 'Team1', 'Team2', 'Venue',
    'TossWinner', 'TossDecision', 'SuperOver', 'WinningTeam', 'WonBy', 'Margin', 'method',
    'Player_of_Match', 'Team1Players', 'Team2Players', 'Umpire1', 'Umpire2', 'Result'
]
columnsToDrop = [
    'WinningTeam', 'ID', 'City', 'Date', 'Season', 'MatchNumber', 'Venue',
    'Player_of_Match', 'Team1Players', 'Team2Players', 'Umpire1', 'Umpire2'
]
matches = matches.drop(columns=columnsToDrop)
columns = matches.columns
for column in columns: matches[column] = label_encoder.fit_transform(matches[column])
X, y = matches[columns[:-1]], matches['Result']

def doClassification():
    for i in range(4):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(0.2 if i%2==0 else 0.1), random_state=1)
        models[0 if i<2 else 1].fit(X_train, y_train)
        y_pred = models[0 if i<2 else 1].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("{} Model Accuracy with {}% test data: {:.2f}%".format(model_names[0 if i<2 else 1], (20 if i%2==0 else 10), accuracy*100))

    k_values = [5, 10]
    for i in range(4):
        overall_conf_matrix = 0
        kf = KFold(n_splits=k_values[i%2], shuffle=True, random_state=1)
        cross_val_results = cross_val_score(models[0 if i<2 else 1], X, y, cv=kf, scoring='accuracy')
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            models[0 if i<2 else 1].fit(X_train, y_train)
            y_pred = models[0 if i<2 else 1].predict(X_test)
            num_classes_fold = len(np.unique(np.concatenate((y_train, y_test)))) 
            fold_conf_matrix = confusion_matrix(y_test, y_pred, labels=range(num_classes_fold))
            if not 'overall_conf_matrix' in locals():
                overall_conf_matrix = np.zeros((num_classes_fold, num_classes_fold), dtype=int)
            overall_conf_matrix += fold_conf_matrix
        print('\n{}-Fold {} results: {}'.format(k_values[i%2], model_names[0 if i<2 else 1], cross_val_results * 100))
        print('Mean accuracy: {:.2f}%'.format(cross_val_results.mean() * 100))
        print('Standard Deviation: {:.2f}'.format(cross_val_results.std()))
        print("Confusion Matrix:\n{}".format(overall_conf_matrix))

    sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Won', 'Lost'], yticklabels=['Won', 'Lost'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()