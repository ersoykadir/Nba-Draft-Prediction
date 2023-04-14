'''
Kadir Ersoy 2018400252
Cmpe 481 Term Project - Nba Draft Prediction

Example usage:
python data_collection.py --target --start --end
python data_collection.py ADVANCED_WS 1990 2005
'''

from basketball_reference_scraper.drafts import get_draft_class
import pandas as pd
from sportsipy.ncaab.teams import Teams
from sportsipy.ncaab.roster import Player
from sportsipy.ncaab.roster import Roster
import time
import numpy as np
import numpy as np

# Gets nba draft data from basketball reference
def get_draft_data(start, end):
    for y in range(start, end):
        data = get_draft_class(y)
        data.to_csv(f'draft{y}.csv')
        print(f'draft{y}.csv', 'saved')
        time.sleep(3.5)

# drop rows with no win shares
def clean_draft_data(start, end):
    for y in range(start, end):
        data = pd.read_csv(f'draft{y}.csv')
        data = data.dropna(axis=0, subset=['ADVANCED_WS'])
        data.to_csv(f'draft{y}_cleaned.csv')

# for every draft player, acquire player name and player id.
def player_data(start_year, end_year):
    teams = pd.read_csv('NCAA_teams.csv')
    teams = pd.DataFrame(teams)
    abbv = teams.loc[:,"abbreviation"]
    for year in range(start_year,end_year):
        rosters = {'year':[],'team_abbv':[],'player_id':[]}
        years = []
        teams = []
        players = []
        player_names = []
        f = open(f"log{year}.txt", "w")
        for team in abbv:
            try:
                team_slim = Roster(team, year = year, slim = True)
            except Exception as e:
                f.write(f"{str(e)}\n")
                continue
            player_ids = team_slim.players.keys()
            for p in player_ids:
                years.append(year)
                teams.append(team)
                players.append(p)
                name = team_slim.players[p]
                player_names.append(name)
            time.sleep(3.5)
            # get team rosters for all teams from 1980s to 2010s
            # team rosters have player ids
            # get player data using those ids
            f.write(f"team:{team} is done!")
            print(f"team:{team} is done!")
        f.write(f"year:{year} is done!")
        print(f"year:{year} is done!")
        f.close()   
        rosters['year'] = years
        rosters['team_abbv'] = teams
        rosters['player_id'] = players
        rosters['player_name'] = player_names
        data = pd.DataFrame(rosters)
        data.to_csv(f'NCAA_players_{year}.csv')

# get ncaa data for each draft player per year
def get_college_data_per_draft_player(year,year_end):
    for y in range(year,year_end):
        players = pd.read_csv(f'draft{y}.csv')
        ncaa_players = pd.read_csv(f'NCAA_players_{y}.csv')
        players = pd.DataFrame(players)
        names = players.loc[:,"PLAYER"]
        colleges = players.loc[:,"COLLEGE"]
        frames = []
        for n,c in zip(names,colleges):
            if pd.isna(c):
                print('player didnot go to college ',n)
                continue
            player = ncaa_players.loc[ncaa_players['player_name'] == n]
            if len(player.index) == 0:
                print('player not found ',n)
                continue
            id = player.loc[:,'player_id'].values[0]
            player_data = Player(id)
            player_data = player_data.dataframe
            player_data.reset_index(inplace=True)
            player_data.rename(columns={'level_0':'season'}, inplace=True)
            college_career = player_data.loc[player_data['season'] == 'Career']
            college_career = pd.DataFrame(college_career)
            college_career['college'] = c
            college_career['player_name'] = n
            college_career['draft_year'] = y
            frames.append(college_career)
            time.sleep(3.5)
        result = pd.concat(frames)
        result.to_csv(f'college_data_draft{y}.csv')

# Prepares data for regression
def get_proper_data(start, end, target):
    clean_data(start, end)
    append_target_value(start, end)
    combine_merged_data(start, end)
    replace_nan_with_mean(start, end, target)

# Cleaning the data from problematic columns
def clean_data(start, end):
    for y in range(start,end):
        data = pd.read_csv(f'college_data_draft{y}.csv')

        # Columns that have more than 50% of the data missing
        data = data.drop('Unnamed: 0', axis=1)
        data = data.drop('season', axis=1)
        data = data.drop('assist_percentage', axis=1)
        data = data.drop('block_percentage', axis=1)
        data = data.drop('box_plus_minus', axis=1)
        data = data.drop('conference', axis=1)
        data = data.drop('defensive_box_plus_minus', axis=1)
        data = data.drop('defensive_rebound_percentage', axis=1)
        data = data.drop('games_played', axis=1)
        data = data.drop('games_started', axis=1)
        data = data.drop('height', axis=1)
        data = data.drop('offensive_box_plus_minus', axis=1)
        data = data.drop('offensive_rebound_percentage', axis=1)
        data = data.drop('steal_percentage', axis=1)
        data = data.drop('total_rebound_percentage', axis=1)
        data = data.drop('usage_percentage', axis=1)
        data = data.drop('weight', axis=1)
        data = data.drop('player_efficiency_rating', axis=1)
        data = data.drop('points_produced', axis=1)
        data = data.drop('win_shares_per_40_minutes', axis=1)
        data = data.drop('win_shares', axis=1)
        data = data.drop('defensive_rebounds', axis=1)
        data = data.drop('offensive_rebounds', axis=1)
        data = data.drop('defensive_win_shares', axis=1)
        data = data.drop('offensive_win_shares', axis=1)
        data = data.drop('free_throw_attempts', axis=1)
        data = data.drop('free_throw_attempt_rate', axis=1)

        data.to_csv(f'college_data_draft_cleaned{y}.csv')

# Combining college data with target value i.e. Win Shares
def append_target_value(start, end):
    frames = []
    for y in range(start, end):
        college_data = pd.read_csv(f'college_data_draft_cleaned{y}.csv')
        college_data = pd.DataFrame(college_data) 

        # remove unnecessary columns
        college_data = college_data.drop(['player_id','team_abbreviation'], axis=1)

        # need to add target column to college_data
        nba_data = pd.read_csv(f'draft{y}_cleaned.csv')
        nba_data = pd.DataFrame(nba_data)
        nba_features = ['PLAYER','TOTALS_G','PICK','ADVANCED_WS', 'ADVANCED_WS/48' ,'ADVANCED_BPM', 'ADVANCED_VORP']
        nba_data = nba_data[nba_features]

        # WS/G, tried to normalize WS but it didn't improve the model
        nba_data['ADVANCED_WS/G'] = nba_data['ADVANCED_WS'] / nba_data['TOTALS_G']
        divisor = (nba_data.index.size) // 2
        nba_data['ROUND'] = 1
        nba_data.loc[divisor:,'ROUND'] = 2
        nba_data.drop('TOTALS_G', axis=1, inplace=True)

        # This was done for classification with 7 classes
        # Players were grouped into 7 classes based on their draft position
        nba_data.loc[nba_data['PICK'] <= 2, 'TARGET'] = 1
        nba_data.loc[(nba_data['PICK'] >= 3) & (nba_data['PICK'] <= 5), 'TARGET'] = 2
        nba_data.loc[(nba_data['PICK'] >= 6) & (nba_data['PICK'] <= 8), 'TARGET'] = 3
        nba_data.loc[(nba_data['PICK'] >= 9) & (nba_data['PICK'] <= 12), 'TARGET'] = 4
        nba_data.loc[(nba_data['PICK'] >= 13) & (nba_data['PICK'] <= 17), 'TARGET'] = 5
        nba_data.loc[(nba_data['PICK'] >= 18) & (nba_data['PICK'] <= 26), 'TARGET'] = 6
        nba_data.loc[(nba_data['PICK'] >= 27) & (nba_data['PICK'] <= 39), 'TARGET'] = 7
        nba_data.loc[(nba_data['PICK'] >= 40) & (nba_data['PICK'] <= 50), 'TARGET'] = 8
        nba_data.loc[nba_data['PICK'] >= 51, 'TARGET'] = 9
        nba_data.drop('PICK', axis=1, inplace=True)
        
        # add to frames
        frames.append(nba_data)

        # merge college_data and nba_data
        combined_data = pd.merge(college_data, nba_data, left_on='player_name', right_on='PLAYER')

        # remove unnecessary columns
        combined_data.drop('PLAYER', axis=1, inplace=True)
        combined_data.drop('Unnamed: 0', axis=1, inplace=True)

        # reorder columns
        first_col = combined_data.pop('draft_year')
        second_col = combined_data.pop('player_name')
        third_col = combined_data.pop('college')
        fourth_col = combined_data.pop('position')
        combined_data.insert(0, 'year', first_col)
        combined_data.insert(1, 'player_name', second_col)
        combined_data.insert(2, 'college', third_col)
        combined_data.insert(3, 'position', fourth_col)
        combined_data.to_csv(f'merged_{y}.csv')

    result = pd.concat(frames)
    # find max advanced_ws player name
    result.reset_index(inplace=True)
    result.drop('index', axis=1, inplace=True)
    index_max_advanced_ws = result['ADVANCED_WS'].argmax()
    result.to_csv(f'nba_success_metrics.csv')

# Combines merged data from all years into one file
def combine_merged_data(start, end):
    frames = []
    for y in range(start, end):
        players = pd.read_csv(f'merged_{y}.csv')
        players = pd.DataFrame(players)
        frames.append(players)
    result = pd.concat(frames)
    result.to_csv(f'merged_{start}-{end}.csv')

# Handling NAN values, also chooses target for regression
def replace_nan_with_mean(start, end, target):
    data = pd.read_csv(f'merged_{start}-{end}.csv')
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('Unnamed: 0.1', axis=1)
    # choosing target
    target_list = ['TARGET','ADVANCED_WS', 'ADVANCED_BPM', 'ADVANCED_VORP', 'ADVANCED_WS/48', 'ADVANCED_WS/G', 'ROUND']
    for t in target_list:
        if t != target:
            data = data.drop(t, axis=1)

    # print average of target
    print('average of target',data[target].mean())

    # replace data with mean for each column
    for c in range(0, len(data.columns)):
        if data.columns[c] == 'year' or data.columns[c] == 'player_name' or data.columns[c] == 'position' or data.columns[c] == 'college':
            continue
        mean = data[data.columns[c]].mean()
        if np.isnan(mean):
            mean = 0
        data[data.columns[c]].fillna(mean, inplace=True)
    data.to_csv(f'merged_{start}-{end}_cleaned.csv')
    print(data.isnull().sum())

# I have stored logs for acquiring data from basketball-reference.com
# This cleans the logs to remove the unsuccessful pulls
def clean_logs():
    for y in range(1990,2019):
        with open(f"log{y}.txt", "r+") as f:
            new_f = f.readlines()
            f.seek(0)
            for line in new_f:
                if 'Can\'t pull' not in line:
                    f.write(line)
            f.truncate()

# Acquires college and nba data for each player in draft class 
def data_preparation(start, end, target):
    get_draft_data(start, end)
    clean_draft_data(start, end)

    player_data(start, end)
    get_college_data_per_draft_player(start, end)

    get_proper_data(start, end, target)

import sys
def main():
    target = sys.argv[1]
    start = sys.argv[2]
    end = sys.argv[3]
    data_preparation(int(start), int(end), target)