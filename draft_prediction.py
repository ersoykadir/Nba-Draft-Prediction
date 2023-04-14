'''
Kadir Ersoy 2018400252
Cmpe 481 Term Project - Nba Draft Prediction

Example usage:
python draft_prediction.py --target --train_start --train_end --test_start --test_end
python draft_prediction.py ADVANCED_WS 1990 2005 2005 2006
'''

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from data_collection import get_proper_data


def parameter_tuning(X_train, y_train, estimator):
    # Create the parameter grid
    param_grid = {
        'max_depth': [2, 4, 6, 8, 10],
        'min_samples_leaf': [2, 4, 6, 8, 10],
        'min_samples_split': [2, 4, 6, 8, 10]
    }

    # Create the grid search object
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv('gridsearch_results.csv')
    print(results)
    return grid_search.best_estimator_

# Plotting real and predicted values, and annotating the players
def plot_res(y_players, y_real, y_pred,fname):
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8)
    ax.scatter(y_players.index, y_real, s=20, edgecolor="black", c="darkorange", label="data_real")
    ax.scatter(y_players.index, y_pred, s=20, edgecolor="black", c="cornflowerblue", label="data_pred")
    counter = 0
    for i in range(y_players.index.start, y_players.index.stop):
        ax.annotate(y_players[i], (i, y_pred[i]),fontsize=8)
        counter += 1
    counter = 0
    for i in range(y_players.index.start, y_players.index.stop):
        ax.annotate(y_players[i], (i, y_real[i]),fontsize=8)
        counter += 1
    ax.legend()
    #plt.show()
    plt.savefig(fname)

# Decision Tree Regression
def regression(target, train_start, train_end, test_start, test_end):
    # get data
    get_proper_data(train_start,train_end, target)
    college_data = pd.read_csv(f'merged_{train_start}-{train_end}_cleaned.csv')
    print(len(college_data.index))
    college_data.describe()
    y_train = college_data[target]
    y_players_train = college_data.player_name
    X_train = college_data.drop([target, 'player_name', 'college', 'position', 'year', 'Unnamed: 0'], axis=1)

    # Create the decision tree regressor
    college_model = DecisionTreeRegressor(random_state=42, max_depth=15)
    college_model.fit(X_train, y_train)

    # Create the decision tree regressor with parameter tuning
    regressor = DecisionTreeRegressor()
    college_model1 = parameter_tuning(X_train, y_train, regressor)
    
    # get test data
    get_proper_data(test_start,test_end, target)
    test = pd.read_csv(f'merged_{test_start}-{test_end}_cleaned.csv')
    y_test = test[target]
    y_players_test = test['player_name']
    X_test = test.drop([target, 'player_name', 'college', 'position', 'year', 'Unnamed: 0'], axis=1)

    # Predict the training set results
    y_prediction_train = college_model.predict(X_train)
    y_prediction_train1 = college_model1.predict(X_train)

    # Training results
    sc = college_model.score(X_train, y_train)
    print(sc, 'train score')
    mse_train = mean_squared_error(y_train, y_prediction_train)
    rmse_train = np.sqrt(mse_train)
    print(mse_train, 'train mse')
    print(rmse_train, 'train rmse')
    adj_r2_train = 1 - (1-sc)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
    print(adj_r2_train, 'train adj r2')

    # Training results with parameter tuning
    sc_1 = college_model1.score(X_train, y_train)
    print(sc_1, 'train score1')
    mse_train1 = mean_squared_error(y_train, y_prediction_train1)
    rmse_train1 = np.sqrt(mse_train1)
    print(mse_train1, 'train mse1')
    print(rmse_train1, 'train rmse1')
    adj_r2_train1 = 1 - (1-sc_1)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
    print(adj_r2_train1, 'train adj r2')
    
    # Predict the test set results
    y_prediction_test = college_model.predict(X_test)
    y_prediction_test1 = college_model1.predict(X_test)

    plot_res(y_players_test, y_test, y_prediction_test, f"DecisionTree-test_{test_start}-{test_end}")
    plot_res(y_players_train, y_train, y_prediction_train, f"DecisionTree-train_{train_start}-{train_end}")

    # Test results
    sc_test = college_model.score(X_test, y_test)
    print(sc_test, 'test score')
    mse_test = mean_squared_error(y_test, y_prediction_test)
    rmse_test = np.sqrt(mse_test)
    print(mse_test, 'test mse')
    print(rmse_test, 'test rmse')
    adj_r2_test = 1 - (1-sc_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    print(adj_r2_test, 'test adj r2')

    # Test results with parameter tuning
    sc_test1 = college_model1.score(X_test, y_test)
    print(sc_test1, 'test score1')
    mse_test1 = mean_squared_error(y_test, y_prediction_test1)
    rmse_test1 = np.sqrt(mse_test1)
    print(mse_test1, 'test mse1')
    print(rmse_test1, 'test rmse1')
    adj_r2_test1 = 1 - (1-sc_test1)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    print(adj_r2_test1, 'test adj r2')

    # plot the training results
    plt.figure(figsize=(14,8),dpi=100)
    plt.scatter(X_train.index, y_train, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_train.index, y_prediction_train, color="cornflowerblue", label="pred", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("ADVANCED_WS")
    plt.title(f"Decision Tree - Training - ADVANCED_WS - {test_start}-{train_end} - RMSE: {rmse_train:.2f} - R2 Score: {sc:.2f}")
    plt.legend()
    plt.savefig('Decision Tree - Training - ADVANCED_WS.png',dpi=100)

    # plot the training results with parameter tuning
    plt.figure(figsize=(14,8),dpi=100)
    plt.scatter(X_train.index, y_train, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_train.index, y_prediction_train1, color="cornflowerblue", label="pred", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("ADVANCED_WS")
    plt.title(f"Decision Tree - Training - ADVANCED_WS - {test_start}-{train_end} - RMSE: {rmse_train1:.2f} - R2 Score: {sc_1:.2f}")
    plt.legend()
    plt.savefig('Decision Tree - Training - ADVANCED_WS - param_tuned.png',dpi=100)

    # plot the testing results
    plt.figure(figsize=(14,8),dpi=100)
    plt.scatter(X_test.index, y_test, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test.index, y_prediction_test, color="cornflowerblue", label="pred", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("ADVANCED_WS")
    plt.title(f"Decision Tree - Test - ADVANCED_WS - {test_start}-{test_end} - RMSE: {rmse_test:.2f} - R2 Score: {sc_test:.2f}")
    plt.legend()
    plt.savefig('Decision Tree - Test - ADVANCED_WS.png',dpi=100)

    # plot the testing results with parameter tuning
    plt.figure(figsize=(14,8),dpi=100)
    plt.scatter(X_test.index, y_test, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test.index, y_prediction_test1, color="cornflowerblue", label="pred", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("ADVANCED_WS")
    plt.title(f"Decision Tree - Test - ADVANCED_WS - {test_start}-{test_end} - RMSE: {rmse_test1:.2f} - R2 Score: {sc_test1:.2f}")
    plt.legend()
    plt.savefig('Decision Tree - Test - ADVANCED_WS - param_tuned.png',dpi=100)


# Parameter Tuning for Random Forest
def parameter_tuning_rf(X_train, y_train, regressor):
    n_estimators = [5,20,50,100] # number of trees in the random forest
    #max_features = ['auto', 'sqrt'] # number of features in consideration at every split
    max_depth = [int(x) for x in np.linspace(10, 120, num = 12)] # maximum number of levels allowed in each decision tree
    min_samples_split = [2, 6, 10] # minimum sample number to split a node
    min_samples_leaf = [1, 3, 4] # minimum sample number that can be stored in a leaf node
    bootstrap = [True, False] # method used to sample data points

    random_grid = {'n_estimators': n_estimators,

    'max_depth': max_depth,

    'min_samples_split': min_samples_split,

    'min_samples_leaf': min_samples_leaf,

    'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator = regressor,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)

    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
    print(rf_random.best_score_)
    return rf_random.best_estimator_

# Random Forest Regression
def random_forest_regression(target, train_start, train_end, test_start, test_end):
    # get the data
    get_proper_data(train_start,train_end, target)
    train = pd.read_csv(f'merged_{train_start}-{train_end}_cleaned.csv')
    y_train = train[target]
    y = train[target]
    y_players_train = train['player_name']
    X_train = train.drop([target, 'player_name', 'college', 'position', 'year', 'Unnamed: 0'], axis=1)
    X = train.drop([target, 'player_name', 'college', 'position', 'year', 'Unnamed: 0'], axis=1)
    
    # Create the Random Forest regressor with parameter tuning
    regressor = RandomForestRegressor()
    college_model1 = parameter_tuning_rf(X_train, y_train, regressor)

    # Create the Random Forest regressor
    college_model = RandomForestRegressor(n_estimators=200)
    college_model.fit(X_train , y_train)

    # get the test data
    get_proper_data(test_start,test_end, target)
    test = pd.read_csv(f'merged_{test_start}-{test_end}_cleaned.csv')
    y_test = test[target]
    y_players_test = test['player_name']
    X_test = test.drop([target, 'player_name', 'college', 'position', 'year', 'Unnamed: 0'], axis=1)

    # Predicting the training set results
    y_prediction_train = college_model.predict(X_train)
    y_prediction_train1 = college_model1.predict(X_train)

    # Training Results
    sc = college_model.score(X_train, y_train)
    print(sc, 'train score')
    mse_train = mean_squared_error(y_train, y_prediction_train)
    rmse_train = np.sqrt(mse_train)
    print(mse_train, 'train mse')
    print(rmse_train, 'train rmse')
    adj_r2_train = 1 - (1-sc)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
    print(adj_r2_train, 'train adj r2')

    # Training Results with parameter tuning
    sc_1 = college_model1.score(X_train, y_train)
    print(sc_1, 'train score1')
    mse_train1 = mean_squared_error(y_train, y_prediction_train1)
    rmse_train1 = np.sqrt(mse_train1)
    print(mse_train1, 'train mse1')
    print(rmse_train1, 'train rmse1')
    adj_r2_train1 = 1 - (1-sc)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
    print(adj_r2_train1, 'train adj r2')

    # Predicting the test set results
    y_prediction_test = college_model.predict(X_test)
    y_prediction_test1 = college_model1.predict(X_test)

    plot_res(y_players_test, y_test, y_prediction_test, f'RandomForest-test_{test_start}-{test_end}.png')
    plot_res(y_players_train, y_train, y_prediction_train, f'RandomForest-train_{train_start}-{train_end}.png')

    # Test Results
    sc_test = college_model.score(X_test, y_test)
    print(sc_test, 'test score')
    mse_test = mean_squared_error(y_test, y_prediction_test)
    rmse_test = np.sqrt(mse_test)
    print(mse_test, 'test mse')
    print(rmse_test, 'test rmse')
    adj_r2_test = 1 - (1-sc_test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    print(adj_r2_test, 'test adj r2')

    # Test Results with parameter tuning
    sc_test1 = college_model1.score(X_test, y_test)
    print(sc_test1, 'test score1')
    mse_test1 = mean_squared_error(y_test, y_prediction_test1)
    rmse_test1 = np.sqrt(mse_test1)
    print(mse_test1, 'test mse1')
    print(rmse_test1, 'test rmse1')
    adj_r2_test1 = 1 - (1-sc_test1)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    print(adj_r2_test1, 'test adj r2')

    # plot the training results
    plt.figure(figsize=(14,8),dpi=100)
    plt.scatter(X_train.index, y_train, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_train.index, y_prediction_train, color="cornflowerblue", label="pred", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("ADVANCED_WS")
    plt.title(f"Random Forest - Training  {target} - {train_start}-{train_end} - RMSE: {rmse_train:.2f} - R2 Score: {sc:.2f}")
    plt.legend()
    plt.savefig('Random Forest - Training - ADVANCED_WS.png',dpi=100)

    # plot the training results with parameter tuning
    plt.figure(figsize=(14,8),dpi=100)
    plt.scatter(X_train.index, y_train, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_train.index, y_prediction_train1, color="cornflowerblue", label="pred", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("ADVANCED_WS")
    plt.title(f"Random Forest - Training - {target} - {train_start}-{train_end} - RMSE: {rmse_train1:.2f} - R2 Score: {sc_1:.2f}")
    plt.legend()
    plt.savefig('Random Forest - Training - ADVANCED_WS - param_tuned.png',dpi=100)

    # plot the testing results
    plt.figure(figsize=(14,8),dpi=100)
    plt.scatter(X_test.index, y_test, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test.index, y_prediction_test, color="cornflowerblue", label="pred", linewidth=2)
    plt.xlabel("data")
    plt.ylabel(target)
    plt.title(f"Random Forest - Test - {target} - {test_start}-{test_end} - RMSE: {rmse_test:.2f} - R2 Score: {sc_test:.2f}")
    plt.legend()
    fname = f'Random Forest - Test - ADVANCED_WS.png'
    plt.savefig(fname,dpi=100)

    # plot the testing results with parameter tuning
    plt.figure(figsize=(14,8),dpi=100)
    plt.scatter(X_test.index, y_test, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test.index, y_prediction_test1, color="cornflowerblue", label="pred", linewidth=2)
    plt.xlabel("data")
    plt.ylabel(target)
    plt.title(f"Random Forest - Test - {target} - {test_start}-{test_end} - RMSE: {rmse_test1:.2f} - R2 Score: {sc_test1:.2f}")
    plt.legend()
    fname = f'Random Forest - Test - ADVANCED_WS - param_tuned.png'
    plt.savefig(fname,dpi=100)

# Pair plotting features with target
def pair_plot(target, start, end):
    get_proper_data(start, end, target)
    data = pd.read_csv(f'merged_{start}-{end}_cleaned.csv')
    data = data.drop('Unnamed: 0', axis=1)
    t = data[target]
    data = data.drop(target, axis=1)
    data[target] = t
    headers = list(data.columns)
    headers = headers[4:-1]

    # draw regplots for each feature in a single figure
    fig, axes = plt.subplots(5, 5, figsize=(18, 10))
    for i, ax in enumerate(axes.flat):
        sns.regplot(x=data[target], y=data[headers[i]], ax=ax) 
    fig.tight_layout()
    fig.savefig('pair_plot.png')

import sys
def main():
    # get year range as arguments
    target = sys.argv[1]
    train_start = int(sys.argv[2])
    train_end = int(sys.argv[3])
    test_start = int(sys.argv[4])
    test_end = int(sys.argv[5])
    regression(target, train_start, train_end, test_start, test_end)
    random_forest_regression(target, train_start, train_end, test_start, test_end)
    pair_plot(target, train_start, train_end)