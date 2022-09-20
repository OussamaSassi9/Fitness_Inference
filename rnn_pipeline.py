import argparse
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GRU, concatenate, Input, LeakyReLU
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import joblib
from sql_queries import *
from connect_db import postgresConnect
from config_app_parser import Parser
from datetime import datetime
import json
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class RnnModel:
    """ """

    def __init__(self) -> None:
        """ """
        parser = Parser()
        model_parameters = parser.get_model_rnn_parameters()
        tune_rnn_parameters = parser.get_tune_rnn_parameters()
        train_rnn_parameters = parser.get_train_parameters()

        self.sequence_step = int(model_parameters["sequence_step"])
        self.min_steps = int(model_parameters["min_steps"])
        self.n_steps_features = int(model_parameters["n_steps_features"])
        self.n_features = int(model_parameters["n_features"])
        self.n_classes = int(model_parameters["n_classes"])
        self.model_dir = model_parameters["model_dir"]
        self.model_label = model_parameters["model_label"]
        self.train_epochs = int(model_parameters["train_epochs"])
        self.batch_size = int(model_parameters["batch_size"])
        self.learning_rate = float(model_parameters["learning_rate"])
        self.class_weight = dict(
            (int(class_id), weight)
            for class_id, weight in json.loads(model_parameters["class_weight"]).items()
        )
        self.max_visit_date = train_rnn_parameters["max_visit_date"]
        self.tune_epochs = int(tune_rnn_parameters["tune_epochs"])
        self.scaler_dir = os.path.join(self.model_dir, "scaler")
        if os.path.isdir(self.scaler_dir):
            # load scaler for shifted features
            self.scaler_shifted = joblib.load(
                os.path.join(self.scaler_dir, "scaler_shifted")
            )

            # load scaler for non shifted features
            self.scaler_nonShifted = joblib.load(
                os.path.join(self.scaler_dir, "scaler_nonShifted")
            )
        else:
            self.scaler_shifted = StandardScaler()
            self.scaler_nonShifted = StandardScaler()

        # connection to database
        postgres_db = postgresConnect()
        self.conn_alchemy = postgres_db.connect_with_sqlalchemy()
        self.conn = postgres_db.connect_with_psycopg2()
        self.cursor = postgres_db.get_psycopgCursor()

    def data_pipeline(self, data, batch_size=32, mode="") -> None:
        """ """
        """
        data.drop_duplicates(
            subset=['client_id', 'visit_date'],
            inplace=True
            )
        """

        drop_features = ["category_id"]
        steps_features = [
            "hour",
            "mov_avg_skipped_days",
            "mov_avg_training_period_minute",
            "mov_avg_training_times_per_week",
            "mov_avg_client_week_discipline",
            "gym_density",
            "attendance_ratio",
        ]

        # encoder = LabelEncoder()
        # data.gender = encoder.fit_transform(data.gender)
        gender_encoding = {"男性": 0, "女性": 1}
        data.gender = data.gender.apply(
            lambda x: gender_encoding[x] if x in ["男性", "女性"] else -1
        )

        if mode == "train":

            # filter clients for training
            filter_mask = data.client_id.value_counts() < self.min_steps
            filter_clients = data.client_id.value_counts()[filter_mask].index
            data = data[~data.client_id.isin(filter_clients)]
            data.reset_index(inplace=True, drop=True)

            # save inference data
            # data[data.client_id.isin(train_clients)]\
            #    .to_csv('inference.csv', index=False)

            # train/validation pipeline
            # fitting scaler on train data
            self.scaler_shifted.fit(
                data.drop(drop_features + ["client_id"], axis=1)[steps_features]
            )
            # save scaler
            os.makedirs(self.scaler_dir, exist_ok=True)
            joblib.dump(
                self.scaler_shifted, os.path.join(self.scaler_dir, "scaler_shifted")
            )

            # setting training sequences and scale data
            grouped_data = data.sort_values(["client_id", "visit_date"]).groupby(
                ["client_id"]
            )

            shifted_train = []
            for i in range(self.sequence_step):
                tmp = (
                    grouped_data.shift(-i)
                    .drop(drop_features, axis=1)[steps_features]
                    .fillna(9999)
                    .values
                )  # shift drop and fillna
                tmp = self.scaler_shifted.transform(tmp)  # scale
                # add step and reshape
                shifted_train.append(tmp.reshape(-1, 1, self.n_steps_features))
            y_train = pd.get_dummies(
                data.sort_values(["client_id", "visit_date"]).category_id
            )

            train_data = data.sort_values(["client_id", "visit_date"])
            train_data = train_data[
                ["age_at_enrollment", "gym_id", "gender", "year", "quarter", "month"]
            ]

            # fit scaler for nonShifted data
            self.scaler_nonShifted.fit(train_data.fillna(2352))
            # save non shifted scaler
            os.makedirs(self.scaler_dir, exist_ok=True)
            joblib.dump(
                self.scaler_nonShifted,
                os.path.join(self.scaler_dir, "scaler_nonShifted"),
            )
            train_data = self.scaler_nonShifted.transform(train_data.fillna(9999))

            # tensorflow data pipeline
            dataset = Dataset.from_tensor_slices(
                ((train_data, tuple(step for step in shifted_train)), y_train)
            )
            dataset = dataset.batch(batch_size=batch_size).shuffle(1000)
            train_data = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            return train_data

        # inference/ fine tuning pipeline
        else:
            # data.to_csv('test.csv', index=False)
            # setting sequences and scale data
            grouped_data = (
                data.sort_values(["client_id", "visit_date"])
                .groupby(["client_id"], as_index=False)
                .shift(-1)[steps_features]
            )

            shifted_data = []
            for i in range(self.sequence_step):
                tmp = grouped_data.shift(-i).reset_index(drop=True)
                tmp["client_id"] = data.client_id
                tmp = (
                    tmp.groupby("client_id")
                    .agg("first")[steps_features]
                    .fillna(2352)
                    .values
                )  # shift drop and fillna
                tmp = self.scaler_shifted.transform(tmp)  # scale
                # add step and reshape
                shifted_data.append(tmp.reshape(-1, 1, self.n_steps_features))

            noshifted_data = data.sort_values(["client_id", "visit_date"])
            # if mode == 'inference':
            noshifted_data = noshifted_data.groupby("client_id").agg("first")
            noshifted_data = noshifted_data[
                ["age_at_enrollment", "gym_id", "gender", "year", "quarter", "month"]
            ]
            noshifted_data = self.scaler_nonShifted.transform(
                noshifted_data.fillna(9999)
            )

            if mode == "inference":
                print(shifted_data)
                y_test = data.category_id
                print("---------", y_test)
                dataset = Dataset.from_tensor_slices(
                    ((noshifted_data, tuple(step for step in shifted_data)), None)
                )
            elif mode == "tune":
                y_train = pd.get_dummies(
                    data.sort_values(["client_id", "visit_date"])
                    .groupby("client_id")
                    .agg("first")
                    .category_id
                )
                dataset = Dataset.from_tensor_slices(
                    ((noshifted_data, tuple(step for step in shifted_data)), y_train)
                )

            dataset = dataset.batch(batch_size=32)
            gen_data = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return gen_data

    def build_model(self) -> None:
        # model architecture keras API
        print("-------------", self.n_features)
        # add features (no steps/ not for rnn)
        input_nosteps = Input(shape=(self.n_features))

        # Dense layer for merged features (DNN + GRU)
        x_nosteps = Dense(81, LeakyReLU(alpha=0.05))(input_nosteps)

        input_steps = []
        # add steps features
        for _ in range(self.sequence_step):
            input_steps.append(Input(shape=(1, self.n_steps_features)))

        # concatenate time step features in order
        x_steps = concatenate(input_steps, axis=1)
        # Lstm / Gru / Rnn
        x_steps = GRU(
            units=128,
            activation="tanh",
            input_shape=(-1, self.sequence_step, self.n_steps_features),
            return_sequences=False,
        )(x_steps)
        x_steps = Dropout(0.3)(x_steps)

        # merge nostep + steps inputs
        x_merged = concatenate([x_steps, x_nosteps], axis=1)

        # output layer
        out = Dense(self.n_classes, activation="softmax")(x_merged)

        model = Model((input_nosteps, input_steps), out)
        # model.summary()
        return model

    def train(self, data) -> None:
        # preprocess data
        print("preprocessing data ..", end="")
        train_data = self.data_pipeline(data, self.batch_size, mode="train")
        print("Done!")

        # build model architecture
        # self.model = self.build_model()
        best_model_path = os.path.join(self.model_dir, "best_rnn_architecture")
        self.model = tf.keras.models.load_model(best_model_path)
        tf.keras.backend.clear_session()

        # compile model
        # optimizer = Adam(learning_rate=self.learning_rate)
        # loss = CategoricalCrossentropy()
        # metrics = [AUC(), CategoricalAccuracy()]
        # self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # fit model

        history = self.model.fit(
            train_data,
            epochs=self.train_epochs,
            shuffle=False,
            workers=-1,
            use_multiprocessing=True,
            # class_weight=self.class_weight
        )

        # save model
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save_weights(os.path.join(self.model_dir, self.model_label))
        return history

    def fineTuning(self, data) -> None:
        # preprocess data
        train_data = self.data_pipeline(data, self.batch_size, mode="tune")
        # build model architecture
        # self.model = self.build_model()
        best_model_path = os.path.join(self.model_dir, "best_rnn_architecture")

        self.model = tf.keras.models.load_model(best_model_path)

        # compile model
        # optimizer = Adam(learning_rate=self.learning_rate)
        # loss = CategoricalCrossentropy()
        # metrics = [AUC(), CategoricalAccuracy()]
        # self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # load weights
        self.model.load_weights(os.path.join(self.model_dir, self.model_label))

        # fit model
        history = self.model.fit(
            train_data,
            epochs=self.tune_epochs,
            shuffle=False,
            workers=-1,
            use_multiprocessing=True,
            class_weight=self.class_weight,
        )

        # save model
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save_weights(os.path.join(self.model_dir, self.model_label))
        return history

    def inference(self, data) -> None:
        # preprocess data
        inference_data = self.data_pipeline(data, mode="inference")
        # y_test = inference_data.category_id
        # build model architecture
        # self.model = self.build_model()
        best_model_path = os.path.join(self.model_dir, "best_rnn_architecture")
        self.model = tf.keras.models.load_model(best_model_path)

        # load model weights
        self.model.load_weights(os.path.join(self.model_dir, self.model_label))

        # model inference
        y_proba = self.model.predict(inference_data)
        # print('------------------inf',inference_data )

        y_pred = np.argmax(self.model.predict(inference_data), axis=1)
        # print('y_prob',np.unique(y_pred) )

        # print(classification_report(y_test, y_pred))

        # data to postgres
        y_proba = pd.DataFrame(y_proba, columns=["proba_0", "proba_1", "proba_2"])
        client_scores = pd.concat(
            [
                data.groupby("client_id", as_index=False).agg("first")["client_id"],
                y_proba,
            ],
            axis=1,
        )
        client_scores["inference_date"] = datetime.now()

        client_scores.to_sql(
            "client_scores", self.conn_alchemy, if_exists="append", index=False
        )
        return y_proba


if __name__ == "__main__":
    rnn_model = RnnModel()
    arg_parser = argparse.ArgumentParser()
    parser = Parser()
    arg_parser.add_argument(
        "-m", "--mode", dest="mode", help="rnn pipeline mode: inference, tune, train"
    )
    args = arg_parser.parse_args()
    mode = args.mode

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
        "gym_density",
        "attendance_ratio",
    ]

    if mode == "train":
        rnn_model.cursor.execute(get_train_data, (rnn_model.max_visit_date,))
        result = rnn_model.cursor.fetchall()
        columns = inference_columns + ["category_id"]
        data = pd.DataFrame(result, columns=columns)
        history = rnn_model.train(data)

    elif mode == "inference":

        inference_parameters = parser.get_inference_parameters()
        gym_labels = eval(inference_parameters["gym_labels"])
        steps = inference_parameters["steps"]

        rnn_model.cursor.execute(get_inference_data, (gym_labels, steps))

        result = rnn_model.cursor.fetchall()

        data = pd.DataFrame(
            result, columns=["row_number"] + inference_columns + ["category_id"]
        )
        data.drop("row_number", axis=1, inplace=True)
        y_proba = rnn_model.inference(data)

    elif mode == "tune":
        tune_parameters = parser.get_tune_rnn_parameters()
        gym_labels = eval(tune_parameters["gym_labels"])
        steps = tune_parameters["steps"]
        max_visit_date = tune_parameters["max_visit_date"]
        rnn_model.cursor.execute(get_tune_data, (gym_labels, max_visit_date, steps))
        result = rnn_model.cursor.fetchall()
        columns = ["row_number"] + inference_columns + ["category_id"]
        data = pd.DataFrame(result, columns=columns)
        data.drop("row_number", axis=1, inplace=True)
        history = rnn_model.fineTuning(data)
