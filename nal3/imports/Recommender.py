import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn import metrics
import math

class Recommender:
    def __init__(self, predictor):
        self.predictor = predictor

    def fit(self, user_data):
        self.predictor.fit(user_data)
        
    def recommend(self, userID, n = 10, rec_seen = True):
        # get ratings of movies
        predictions = self.predictor.predict(userID)
        # if no predictions, return
        if len(predictions) == 0:
            return
        predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        invalid_predictions = None
        # if user rated (watched) the movie
        if rec_seen:
            # find only rated movied
            invalid_predictions = self.predictor.data.loc[self.predictor.data['userID'] == userID]['movieID'].to_numpy()
            # loop through all predictions and find valid ones
            for key in invalid_predictions:
                predictions[key] = 0
        # sort predictions by rating
        # init ouput array
        output = []
        predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        for key in predictions:
            # if output is full, stop finding good movies
            if len(output) == n:
                break
            output.append((key, predictions[key]))
        return output

    def evaluate(self, user_id, test_set):
        predictions = self.predictor.predict(user_id)
        test_set_movie_ids = test_set['movieID'].unique()
        
        prediction_values = {}
        test_set_values = {}
        for movie_id in test_set_movie_ids:
            test_set_values[movie_id] = test_set.loc[test_set['movieID'] == movie_id]['rating'].to_numpy()[0] 
            prediction_values[movie_id] = round(predictions[movie_id] * 2) / 2

        X = list(prediction_values.values())
        Y = list(test_set_values.values())

        mae, rmse, recall, accuraccy, f1 = 0, 0, 0, 0, 0
        TP, FP, TN, FN = 0, 0, 0,0
        # lahko das (> 3) => (>= 3) in spustiš mejo kaj je positive
        for i in range(len(X)):
            if X[i] > 3:
                if Y[i] > 3:
                    TP += 1
                else: 
                    FP += 1
            else:
                if Y[i] > 3:
                    FN += 1
                else:
                    TN += 1
        # calc mae
        mae = self.mae(X, Y)
        # calc rmse
        mse = metrics.mean_squared_error(Y, X)
        rmse = math.sqrt(mse)
        # calc recall
        recall = TP / (TP + FN)
        # calc accuraccy
        accuraccy = (TP + TN) / (TP + TN + FP + FN)
        # calc f1
        f1 = (2 * TP) / (2 * TP + FP + FN)
        return mae, rmse, recall, accuraccy, f1

    def mae(self, pred_list, true_list):
        if len(pred_list) != len(true_list):
            raise Exception('Error: number of elements not match!')
        return sum(map(lambda t: float(t[0] - t[1]), zip(pred_list, true_list))) / len(true_list)