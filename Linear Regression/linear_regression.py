
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df = pd.read_csv("football_players.csv")

df = df.drop(['Name',
              'Nationality',
              'National_Position',
              'National_Kit',
              'Club',
              'Club_Kit',
              'Club_Position',
              'Club_Joining',
              'Contract_Expiry',
              'Height',
              'Weight',
              'Preffered_Foot',
              'Work_Rate',
              'Birth_Date',
              'Age',
              'Acceleration',
              'Preffered_Position',
              'DefensiveText',
              'AttackingText',
              'Attacking',
              'Defensive'
              ], axis =1)

columns = [ 'Weak_foot',
            'Skill_Moves',
            'Ball_Control',
            'Dribbling',
            'Marking',
            'Sliding_Tackle',
            'Standing_Tackle',
            'Aggression',
            'Reactions',
            'Attacking_Position',
            'Interceptions',
            'Vision',
            'Composure',
            'Crossing',
            'Short_Pass',
            'Long_Pass',
            'Speed',
            'Stamina',
            'Strength',
            'Balance',
            'Agility',
            'Jumping',
            'Heading',
            'Shot_Power',
            'Finishing',
            'Long_Shots',
            'Curve',
            'Freekick_Accuracy',
            'Penalties',
            'Volleys',
            'GK_Positioning',
            'GK_Diving',
            'GK_Kicking',
            'GK_Handling',
            'GK_Reflexes']

labels = df['Rating'].values
features = df[list(columns)].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)

regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)

Accuracy = regr.score(X_train, y_train)
print ("Accuracy in the training data: ", Accuracy*100 , "%")

accuracy = regr.score(X_test, y_test)
print ("Accuracy in the test data", accuracy*100, "%")