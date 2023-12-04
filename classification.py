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
decisionTreeClassifier = DecisionTreeClassifier()
randomForestClassifier = RandomForestClassifier(n_estimators=5, random_state=42)

columns = ['TossWinner', 'TossDecision', 'SuperOver', 'Team1', 'Team2', 'WonBy', 'Margin', 'WinningTeam']
matches = pd.read_csv(".\\datasets\\IPL_Matches_2008_2022.csv")[columns]

for column in columns: matches[column] = label_encoder.fit_transform(matches[column])
X, y = matches[columns[:-1]], matches['WinningTeam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decisionTreeClassifier.fit(X_train, y_train)
y_pred = decisionTreeClassifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Model Accuracy with 20% test data: {:.2f}%".format(accuracy*100))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
decisionTreeClassifier.fit(X_train, y_train)
y_pred = decisionTreeClassifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Model Accuracy with 10% test data: {:.2f}%".format(accuracy*100))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
randomForestClassifier.fit(X_train, y_train)
y_pred = randomForestClassifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy with 20% test data: {:.2f}".format(accuracy*100))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
randomForestClassifier.fit(X_train, y_train)
y_pred = randomForestClassifier.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy with 10% test data: {:.2f}".format(accuracy*100))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results = cross_val_score(randomForestClassifier, X, y, cv=kf, scoring='accuracy')
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    randomForestClassifier.fit(X_train, y_train)
    y_pred = randomForestClassifier.predict(X_test)

    num_classes_fold = len(np.unique(np.concatenate((y_train, y_test)))) 
    fold_conf_matrix = confusion_matrix(y_test, y_pred, labels=range(num_classes_fold))

    if not 'overall_conf_matrix' in locals():
        overall_conf_matrix = np.zeros((num_classes_fold, num_classes_fold), dtype=int)
    overall_conf_matrix += fold_conf_matrix
print('5-Fold Cross-validation results: {}%'.format(cross_val_results * 100))
print('Mean accuracy: {}%'.format(cross_val_results.mean() * 100))
print('Standard Deviation: {:.2f}'.format(cross_val_results.std()))
# print("Confusion Matrix: {}\n".format(overall_conf_matrix))


kf = KFold(n_splits=10, shuffle=True, random_state=42)
cross_val_results = cross_val_score(randomForestClassifier, X, y, cv=kf, scoring='accuracy')
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    randomForestClassifier.fit(X_train, y_train)
    y_pred = randomForestClassifier.predict(X_test)

    num_classes_fold = len(np.unique(np.concatenate((y_train, y_test)))) 
    fold_conf_matrix = confusion_matrix(y_test, y_pred, labels=range(num_classes_fold))

    if not 'overall_conf_matrix' in locals():
        overall_conf_matrix = np.zeros((num_classes_fold, num_classes_fold), dtype=int)
    overall_conf_matrix += fold_conf_matrix
print('5-Fold Cross-validation results: {}'.format(cross_val_results * 100))
print('Mean accuracy: {}%'.format(cross_val_results.mean() * 100))
print('Standard Deviation: {:.2f}'.format(cross_val_results.std()))
# print("Confusion Matrix: {}\n".format(overall_conf_matrix))

# sns.heatmap(overall_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=columns, yticklabels=columns)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()