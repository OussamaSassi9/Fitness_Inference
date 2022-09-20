"""
@Rutilea
A module that executes a pipeline to train machine learning models using 
gridsearch and stratified KFold.
"""

from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sql_queries import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from connect_db import postgresConnect
from config_app_parser import Parser
import joblib
import os
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import ndcg_score


class Machine_Learning_Model:
    """
    A class that manages mahcine learning models training and
    predicting the clients scores.

    '''

    Attributes
    ----------
    lag : int
        An integer object representing the number of lags
        for a specific set of features
    start_lag : int
        An integer object representing the i-th lag for
        a specific set of features to start lagging.
    model_dir : str
        The directory where to save the models
    model_label : str
        An str object representing the model label
    train_models : str
        An str object containing the models to be trained
    conn_alchemy : object
        A connection instance of the sqlalchemy library.
    conn : object
        A connection instance of psycopg2 library.
    cursor : object
        A cursor object of the psycopg2 library used to
        create, update and delete data from the database.

    Methods
    -------
    data_pipeline(data):
        A function that generates lag features for the machine learning models
        and returns the new generated data.
    inference(data):
        A function that predicts the clients scores using the machine learning
        models and insert them in the postgres database.
    train(data):
        A function that trains the model and save it in a specific folder.
    """

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes for the Machine_Learning_Model
        object.
        """

        parser = Parser()
        machine_learning_parameters = parser.get_machine_learning_parameters()
        self.lag = int(machine_learning_parameters["lag"])
        self.start_lag = int(machine_learning_parameters["start_lag"])
        self.model_dir = machine_learning_parameters["model_dir"]
        self.model_label = machine_learning_parameters["model_label"]
        self.train_models = machine_learning_parameters["train_models"]

        # connection to database
        postgres_db = postgresConnect()
        self.conn_alchemy = postgres_db.connect_with_sqlalchemy()
        self.conn = postgres_db.connect_with_psycopg2()
        self.cursor = postgres_db.get_psycopgCursor()

    def __save_model(self, grid_cv):
        # fit model with best parameters and save it
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(
            grid_cv.best_estimator_, os.path.join(self.model_dir, self.model_label)
        )

    def data_pipeline(self, data):
        """
        A function that generates lag features for the machine learning models
        and returns the new generated data.

        Parameters
        ----------
        data : DataFrame
            A DataFrame object containing the features needed for the
            data pipeline to be executed.

        Returns
        -------
        data : DataFrame
            A DataFrame object containing the transformed data.
        """

        for i in range(self.lag):
            data.loc[
                :, "training_period_minute_lag{}".format(i + self.start_lag)
            ] = data.groupby("client_id").mov_avg_training_period_minute.shift(
                i + self.start_lag
            )

        for i in range(self.lag):
            data.loc[
                :, "mov_avg_client_week_discipline_lag{}".format(i + self.start_lag)
            ] = data.groupby("client_id").mov_avg_client_week_discipline.shift(
                i + self.start_lag
            )

        for i in range(self.lag):
            data.loc[:, "skipped_days_lag{}".format(i + self.start_lag)] = data.groupby(
                "client_id"
            ).mov_avg_skipped_days.shift(i + self.start_lag)
        """
         for i in range(self.lag):
            data.loc[
                :, 'athelic_lag{}'.format(i+self.start_lag)
                ] = data.groupby('client_id')\
                    .athelic\
                    .shift(i+self.start_lag)
        """

        gender_encoding = {"男性": 0, "女性": 1}
        data.gender = data.gender.apply(
            lambda x: gender_encoding[x] if x in ["男性", "女性"] else -1
        )

        splitby_client_target = data[["client_id", "category_id"]].drop_duplicates()
        X_train, X_test, y_train, y_test = train_test_split(
            splitby_client_target.drop("category_id", axis=1),
            splitby_client_target.category_id,
            stratify=splitby_client_target.category_id,
            test_size=0.2,
            random_state=2022,
        )

        train_clients = X_train.client_id.values
        test_clients = X_test.client_id.values

        data.replace(np.inf, np.nan, inplace=True)
        data.fillna(0, inplace=True)
        data.year.unique()

        X_train = data[data.client_id.isin(train_clients)].drop("category_id", axis=1)
        y_train = data[data.client_id.isin(train_clients)].category_id
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        X_test = data[data.client_id.isin(test_clients)].drop("category_id", axis=1)
        y_test = data[data.client_id.isin(test_clients)].category_id

        return data, X_train, y_train, X_test, y_test

    def train(self, data) -> None:
        """
        A function that trains machine learining models using gridsearch
        and stratified KFold and save them in a specific folder.

        Parameters
        ----------
        data : DataFrame
            A DataFrame object containing the features needed for the
            data pipeline to be executed.

        Returns
        -------
        None

        """

        data, X_train, y_train, X_test, y_test = self.data_pipeline(data)
        drop_features = ["visit_date", "client_id"]

        if "lgbm" in self.train_models:
            lgbm = LGBMClassifier()
            param_grid = {
                "learning_rate": [0.001, 0.005, 0.01],
                "n_estimators": [300, 500, 700],
                #'boosting_type' : ['gbdt', 'dart'],
                "colsample_bytree": [0.75, 0.85, 0.95],
                "subsample": [0.8, 0.9],
                #'reg_alpha' : [1, 1.2],
                #'reg_lambda' : [1, 1.2, 1.4],
                "max_depth": [3, 4, 5, 6],
            }
            grid_cv = GridSearchCV(
                verbose=10,
                estimator=lgbm,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=5),
                scoring="f1_macro",
                refit=True,
            )

            grid_cv.fit(X_train.drop(drop_features, axis=1), y_train)
            print(
                "best mean cross-validation score: {:.3f}".format(grid_cv.best_score_)
            )
            print("best parameters:", grid_cv.best_params_)

            y_pred = grid_cv.best_estimator_.predict(X_test.drop(drop_features, axis=1))
            predictions = [round(value) for value in y_pred]
            # evaluate predictions
            accuracy = accuracy_score(y_test, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            print(confusion_matrix(y_test, y_pred))
            target_names = ["class 0", "class 1", "class 2"]
            print(classification_report(y_test, y_pred, target_names=target_names))

            # fit model and save it
            self.__save_model(grid_cv)

        if "rf" in self.train_models:
            rf = RandomForestClassifier()
            param_grid = {
                "n_estimators": [200, 500],
                "max_features": ["sqrt", "log2"],
                "max_depth": [4, 5, 6, 7, 8],
                "criterion": ["gini", "entropy"],
            }
            grid_cv = GridSearchCV(
                verbose=10,
                estimator=rf,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=5),
                scoring="f1_macro",
                refit=True,
            )
            grid_cv.fit(X_train.drop(drop_features, axis=1), y_train)
            print(
                "best mean cross-validation score: {:.3f}".format(grid_cv.best_score_)
            )
            print("best parameters:", grid_cv.best_params_)

            y_pred = grid_cv.best_estimator_.predict(X_test.drop(drop_features, axis=1))
            predictions = [round(value) for value in y_pred]
            # evaluate predictions
            accuracy = accuracy_score(y_test, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            print(confusion_matrix(y_test, y_pred))
            target_names = ["class 0", "class 1", "class 2"]
            print(classification_report(y_test, y_pred, target_names=target_names))

            # fit model and save it
            self.__save_model(grid_cv)

        if "xgb" in self.train_models:
            xgb = XGBClassifier()

            parameters = {
                "nthread": [4],  # when use hyperthread, xgboost may become slower
                "objective": ["binary:logistic"],
                #'learning_rate': [0.03,0.3,0.1,0.05], #so called `eta` value
                "learning_rate": [0.03, 0.3, 0.1],
                "max_depth": [2, 6, 12],
                #'min_child_weight': [5,11,15],
                #'subsample': [0.5,0.75,0.8,1],
                #'colsample_bytree': [0.5,0.7,1],
                #'n_estimators': [100], #number of trees, change it to 1000 for better results
                #'missing':[-999],
                #'seed': [1337] ,
                "weights": [{0: 1, 1: 5, 2: w} for w in [20]],
            }
            # class_weights = class_weight.compute_class_weight(class_weight ='balanced',
            #                                   classes = np.unique(data.category_id),
            #                                  y= data.category_id)
            # class_weights = dict(zip(np.unique(data.category_id), class_weights))

            grid_cv = GridSearchCV(
                verbose=10,
                estimator=xgb,
                param_grid=parameters,
                cv=StratifiedKFold(n_splits=5),
                scoring="f1_macro",
                refit=True,
            )
            grid_cv.fit(X_train.drop(drop_features, axis=1), y_train)
            print(
                "best mean cross-validation score: {:.3f}".format(grid_cv.best_score_)
            )
            print("best parameters:", grid_cv.best_params_)

            y_pred = grid_cv.best_estimator_.predict(X_test.drop(drop_features, axis=1))
            predictions = [round(value) for value in y_pred]
            # evaluate predictions
            accuracy = accuracy_score(y_test, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
            print(confusion_matrix(y_test, y_pred))
            target_names = ["class 0", "class 1", "class 2"]
            print(classification_report(y_test, y_pred, target_names=target_names))

            # fit model and save it
            self.__save_model(grid_cv)

    def inference(self, data) -> None:
        """
        A function that uses a machine learining model
        (Random Forest or XGBoost or LighGBM) to predict the clients
        scores.

        Parameters
        ----------
        data : DataFrame
            A DataFrame object containing the features needed for the
            data pipeline to be executed.

        Returns
        -------
        None

        """

        data, X_train, y_train, X_test, y_test = self.data_pipeline(data)
        drop_features = ["visit_date", "client_id", "category_id"]
        data = data.groupby("client_id", as_index=False).agg("first")
        model = joblib.load(os.path.join(self.model_dir, self.model_label))
        y_proba = model.predict_proba(data.drop(drop_features, axis=1))

        # data to postgres
        y_proba = pd.DataFrame(y_proba, columns=["proba_0", "proba_1", "proba_2"])

        client_scores = pd.concat([data["client_id"], y_proba], axis=1)
        client_scores["inference_date"] = datetime.now()

        client_scores.to_sql(
            "client_scores", self.conn_alchemy, if_exists="append", index=False
        )


if __name__ == "__main__":
    ML_model = Machine_Learning_Model()
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m", "--mode", dest="mode", help="rnn pipeline mode: inference, train"
    )
    args = arg_parser.parse_args()
    mode = args.mode
    parser = Parser()

    inference_columns = [
        "client_id",
        "visit_date",
        "age_at_enrollment",
        "gender",
        "hour",
        "year",
        "quarter",
        "month",
        "gym_id",
        "mov_avg_skipped_days",
        "mov_avg_training_period_minute",
        "mov_avg_training_times_per_week",
        "mov_avg_client_week_discipline",
        "attendance_ratio",
        "gym_density",
    ]

    if mode == "train":
        train_parameters = parser.get_train_parameters()
        max_visit_date = train_parameters["max_visit_date"]

        ML_model.cursor.execute(get_train_data, (max_visit_date,))
        result = ML_model.cursor.fetchall()
        columns = inference_columns + ["category_id"]
        data = pd.DataFrame(result, columns=columns)
        ML_model.train(data)

    elif mode == "inference":
        inference_parameters = parser.get_inference_parameters()
        gym_labels = eval(inference_parameters["gym_labels"])
        steps = inference_parameters["steps"]

        ML_model.cursor.execute(get_inference_data, (gym_labels, steps))
        result = ML_model.cursor.fetchall()
        data = pd.DataFrame(
            result, columns=["row_number"] + inference_columns + ["category_id"]
        )
        data.drop("row_number", axis=1, inplace=True)
        y_proba = ML_model.inference(data)
