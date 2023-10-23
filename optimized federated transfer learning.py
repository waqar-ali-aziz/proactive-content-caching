import concurrent.futures
import warnings
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

import subprocess
import os
import numpy as np
import csv
import multiprocessing

warnings.filterwarnings("ignore")


class BSExecutor:
    def __init__(self, id, num_execute, param_grid, X_train, X_test, y_train, y_test, DS):
        self.id = id
        self.num_execute = num_execute
        self.param_grid = param_grid
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.DS = DS

    def run_command_program(self, results):
        xgb_model = XGBClassifier()

        grid_search = GridSearchCV(estimator=xgb_model, param_grid=self.param_grid, scoring='accuracy', cv=3)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)

        results["result " + str(self.id)] = {'id': self.id, 'best_params': best_params, 'accuracy': accuracy,
                                             'confusion_matrix': cm, 'DS': self.DS}


def createConfigurationsFile(filename, numberof_estimators, numberof_max_depth, array_with_learning_rate):
    numberofRates = len(array_with_learning_rate)
    numberofLines = numberof_estimators * numberof_max_depth * numberofRates
    estimators = range(1, numberof_estimators + 1)
    max_depth = range(1, numberof_max_depth + 1)
    arrayOfData = []
    x = -1
    extracount = 0
    for i in range(0, numberofLines):
        extracount = i % (numberof_estimators)
        if extracount == 0:
            x = (x + 1) % (numberof_estimators)
        else:
            extracount = extracount + 1
        row = [i + 1, estimators[x], max_depth[i % (numberof_max_depth)], array_with_learning_rate[i % (numberofRates)], 1, "None"]
        arrayOfData.append(row)
    header = ['id', 'n_estimator', 'max_depth', 'learning_rate', 'Performance_metric', 'DS']
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in arrayOfData:
            writer.writerow(row)
    return


def loadDatasourceFromFile(filename):
    dataset = pd.read_csv(filename).to_numpy()
    X = dataset[:, 0:10]
    Y = dataset[:, 10]
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def createResultsFile(filename, arrayOfResults):
    fields = ['id', 'n_estimator', 'max_depth', 'learning_rate', 'Performance_metric', "confusion_matrix", "DS"]
    with open(filename, 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in arrayOfResults:
            writer.writerow(row)
    return


def process_job(id, count, param_grid, train_data, results):
    executor = BSExecutor(id, count, param_grid, train_data["X_train"],
                          train_data["X_test"], train_data["y_train"],
                          train_data["y_test"], train_data["name_of_DS"])
    executor.run_command_program(results)


def main():
    filename = 'Configuration_ds.csv'
    path = './' + filename

    if not os.path.isfile(path):
        array_with_learning_rate = [0.1]
        createConfigurationsFile(filename, 30, 30, array_with_learning_rate)

    Configurationdataset = pd.read_csv('Configuration_ds.csv').to_numpy()
    train_arr = []
    countDS = 0
## book is the dataset, with book0 for base station 1, book1 for base station 2, and book2 for base station 3. the dataset is also provided. 
    for index in range(0, 3):
        X_train, X_test, y_train, y_test = loadDatasourceFromFile('book' + str(index) + '.csv')
        train_grid = {
            'ID': index,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'name_of_DS': 'book' + str(index) + '.csv'
        }
        countDS = countDS + 1
        train_arr.append(train_grid)

    param_grids = []
    count = 1
    param_grids = []
    param_grids_support = []

    for row in Configurationdataset:
        param_grid = {'n_estimators': [int(row[1])], 'max_depth': [int(row[2])], 'learning_rate': [row[3]]}
        param_grid_support = {'ID': [int(row[0])], 'Performance_metric': [row[4]]}
        param_grids.append(param_grid)
        param_grids_support.append(param_grid_support)
        count = count + 1

    print("Configurations for " + str(count) + " number of BS")
    print(param_grids)

    results = []
    id = 0
    manager = multiprocessing.Manager()
    results = manager.dict()
    processes = []

    for param_grid in param_grids:
        processes.append(
            (id, count, param_grid, train_arr[id % countDS], results))
        id = id + 1

    print("Jobs Submitted")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(lambda args: process_job(*args), processes)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")

    param_grid_results = []
    counter = 0
    print(results)

    for program, result in results.items():
        print(f"{program}:")
        print("Best Parameters:", result['best_params'])
        print("Accuracy:", result['accuracy'])


    for program, result in results.items():
        param_grid = {
            'id': f"{result['id']}",
            'n_estimator': result['best_params']['n_estimators'],
            'max_depth': result['best_params']['max_depth'],
            'learning_rate': result['best_params']['learning_rate'],
            'Performance_metric': result['accuracy'],
            'confusion_matrix': result['confusion_matrix'],
            'DS': result['DS'],
        }
        param_grid_results.append(param_grid)
        print(result)
        print(f"{program}:")
        print("Best Parameters:", result['best_params'])
        print("Accuracy:", result['accuracy'])

    createResultsFile("results.csv", param_grid_results)


if __name__ == "__main__":
    main()
