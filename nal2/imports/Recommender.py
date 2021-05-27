import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime

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


class RecommenderTopSimilar:
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
        predictions = dict(sorted(predictions.items(), key=lambda item: item[1][1], reverse=True))
        for key in predictions:
            # if output is full, stop finding good movies
            if len(output) == n:
                break
            output.append((key, predictions[key]))
        return output

