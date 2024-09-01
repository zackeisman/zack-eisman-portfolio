#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:03:44 2023

@author: maddiecoe
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# production
data = pd.read_csv('2223_ushl_stats.csv', encoding='latin-1')

features = ['G', 'A', 'PTS']
X = data[features]

new_Y = ['Expected Goals For WOI']
y = data[new_Y]

data = data.dropna(subset=['Expected Goals For WOI'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

production_model = xgb.XGBRegressor()

production_model.fit(X_train, y_train)

production_war_predictions = production_model.predict(X_test)

mae_production_war = mean_absolute_error(y_test, production_war_predictions)
print(f'Mean Absolute Error: {mae_production_war:.2f}')

player_war = []

for index, player_row in data.iterrows():
    player_data = player_row[features].values.reshape(1, -1)
    player_name = player_row['Player']
    player_production_war = production_model.predict(player_data)[0]
    player_war.append((player_name, player_production_war))

war_df = pd.DataFrame(player_war, columns=['Player', 'Predicted Production WAR'])

data = data.merge(war_df, on='Player', how='left')

predicted_production_war = data['Predicted Production WAR']

scaler = MinMaxScaler(feature_range=(0, 100))
data['Scaled Production WAR'] = scaler.fit_transform(predicted_production_war.values.reshape(-1, 1))


# offense
features = ['S', 'Successful Pass to Slot For WOI', 'PDP/20', 'OGP/20']
X = data[features]

new_Y = ['Expected Goals For WOI']
y = data[new_Y]

data = data.dropna(subset=['Expected Goals For WOI'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

offense_model = xgb.XGBRegressor()

offense_model.fit(X_train, y_train)

offense_war_predictions = offense_model.predict(X_test)

mae_offense_war = mean_absolute_error(y_test, offense_war_predictions)
print(f'Mean Absolute Error: {mae_offense_war:.2f}')

player_war = []

for index, player_row in data.iterrows():
    player_data = player_row[features].values.reshape(1, -1)
    player_name = player_row['Player']
    player_offense_war = offense_model.predict(player_data)[0]
    player_war.append((player_name, player_offense_war))

war_df = pd.DataFrame(player_war, columns=['Player', 'Predicted Offense WAR'])

data = data.merge(war_df, on='Player', how='left')

predicted_offense_war = data['Predicted Offense WAR']

scaler = MinMaxScaler(feature_range=(0, 100))
data['Scaled Offense WAR'] = scaler.fit_transform(predicted_offense_war.values.reshape(-1, 1))

# defense
features = ['+/-', 'Shot Attempts Against WOI', 'Shots On Net Against WOI', 'DZ Rebound Recovery Rate WOI']
X = data[features]

new_Y = ['Expected Goals Against WOI']
y = data[new_Y]

data = data.dropna(subset=['Expected Goals Against WOI'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

defense_model = xgb.XGBRegressor()

defense_model.fit(X_train, y_train)

defense_war_predictions = defense_model.predict(X_test)

mae_defense_war = mean_absolute_error(y_test, defense_war_predictions)
print(f'Mean Absolute Error: {mae_defense_war:.2f}')

player_war = []

for index, player_row in data.iterrows():
    player_data = player_row[features].values.reshape(1, -1)
    player_name = player_row['Player']
    player_defense_war = defense_model.predict(player_data)[0]
    player_war.append((player_name, player_defense_war))

war_df = pd.DataFrame(player_war, columns=['Player', 'Predicted Defense WAR'])

data = data.merge(war_df, on='Player', how='left')

predicted_defense_war = data['Predicted Defense WAR']

scaler = MinMaxScaler(feature_range=(0, 100))
data['Scaled Defense WAR'] = scaler.fit_transform(predicted_defense_war.values.reshape(-1, 1))

#transition
features = ['Carry-Outs with Play After', 'Total Carry-Out Attempts', 'Pass-Outs with Play After', 'Controlled Exits', 'Successful Dump Out Attempts']
X = data[features]

carryOut = data['Carry-Out with Play After Rate'].astype(float)  
passOut = data['Pass-Out with Play After Rate'].astype(float)  
controlledExit = data['Controlled Exit with Play After Rate'].astype(float) 
dumpOut = data['Dump Out Success Rate'].astype(float) 

data = data.dropna(subset=['Carry-Out with Play After Rate', 'Pass-Out with Play After Rate', 'Controlled Exit with Play After Rate', 'Dump Out Success Rate'])
X = data[features]
carryOut = data['Carry-Out with Play After Rate']
passOut = data['Pass-Out with Play After Rate']
controlledExit = data['Controlled Exit with Play After Rate']
dumpOut = data['Dump Out Success Rate']

newTarget = (carryOut + passOut + controlledExit + dumpOut) / 4

data['newTarget'] = newTarget

X_train, X_test, y_train, y_test = train_test_split(X, data['newTarget'], test_size=0.2, random_state=42)

transition_model = xgb.XGBRegressor()

transition_model.fit(X_train, y_train)

transition_war_predictions = transition_model.predict(X_test)

mae_transition_war = mean_absolute_error(y_test, transition_war_predictions)
print(f'Mean Absolute Error: {mae_transition_war:.2f}')

player_war = []

for index, player_row in data.iterrows():
    player_data = player_row[features].values.reshape(1, -1)
    player_name = player_row['Player']
    player_transition_war = transition_model.predict(player_data)[0]
    player_war.append((player_name, player_transition_war))

war_df = pd.DataFrame(player_war, columns=['Player', 'Predicted Transition WAR'])

data = data.merge(war_df, on='Player', how='left')

predicted_transition_war = data['Predicted Transition WAR']

scaler = MinMaxScaler(feature_range=(0, 100))
data['Scaled Transition WAR'] = scaler.fit_transform(predicted_transition_war.values.reshape(-1, 1))

# Create an empty array for overall scores
overall_scores = np.zeros(len(data))

# Loop through each unique player in the DataFrame
for player_name in data['Player'].unique():
    player_rows = data[data['Player'] == player_name]
    
    weight_production = 0
    weight_offense = 0
    weight_defense = 0
    weight_transition = 0
    
    # Check the player's position
    player_position = player_rows['Position'].iloc[0]  # Assuming the position is the same for all rows of a player
    if player_position == "F":  # Assuming 'F' denotes an offensive player
        weight_production = 0.4
        weight_offense = 0.4
        weight_defense = 0.1
        weight_transition = 0.1
    elif player_position == "D":  # Assuming 'D' denotes a defensive player
        weight_production = 0.1
        weight_offense = 0.1
        weight_defense = 0.7
        weight_transition = 0.1
    
    # Calculate the overall score for the current player
    overall_score = (
        weight_production * player_rows['Scaled Production WAR'].iloc[0] +
        weight_offense * player_rows['Scaled Offense WAR'].iloc[0] +
        weight_defense * player_rows['Scaled Defense WAR'].iloc[0] +
        weight_transition * player_rows['Scaled Transition WAR'].iloc[0]
    )
    
    # Assign the overall score to all rows corresponding to the current player
    overall_scores[player_rows.index] = overall_score

# Add the overall scores to your DataFrame
data['Overall_Score'] = overall_scores

# Print the names and overall scores
scaled_values = (data[['Player', 'Overall_Score']])
print(scaled_values.drop_duplicates().to_string(index=False))

# Plot the distribution of Overall_Score using Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data['Overall_Score'], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of Overall Scores')
plt.xlabel('Overall Score')
plt.ylabel('Frequency')
plt.show()