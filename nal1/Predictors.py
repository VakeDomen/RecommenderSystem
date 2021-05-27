import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime

class RandomPredictor:
    def __init__(self, min_rating, max_rating):
        self.min_rating = min_rating
        self.max_rating = max_rating

    def fit(self, user_item_data):
        self.data = user_item_data.data
        

    def predict(self, user_id):
        # from DataFrame column of movie ids to array
        modvieIds = self.data['movieID'].to_numpy()
        # init dict object
        movieDict = {}
        # loop through movie ids
        for id in tqdm(modvieIds):
            # generate random number for each id
            movieDict[id] = random.randint(self.min_rating, self.max_rating)
        return movieDict

class ViewsPredictor:
    def fit(self, user_item_data):
        self.data = user_item_data.data
        movieIds = list(dict.fromkeys(self.data['movieID'].to_numpy()))
        # sum of all rating for a movie { movieID: (vs, n) }
        self.movie_dict = {}   
        for id in tqdm(movieIds):
            ratings = self.data.loc[self.data['movieID'] == id]['rating']
            self.movie_dict[id] = len(ratings)

    def predict(self, user_id):
        return self.movie_dict

class AveragePredictor:
    def __init__(self, b=0):
        self.b = max(0, b)

    def fit(self, user_item_data):
        self.data = user_item_data.data
        movieIds = list(dict.fromkeys(self.data['movieID'].to_numpy()))
        # sum of all rating for a movie { movieID: (vs, n) }
        stats = {}
        total_stats = [0, 0]   
        for id in tqdm(movieIds):
            ratings = self.data.loc[self.data['movieID'] == id]['rating']
            stats[id] = [ratings.sum(), len(ratings)]
            total_stats[0] = total_stats[0] + stats[id][0]
            total_stats[1] = total_stats[1] + stats[id][1]
        # init dict object
        movie_dict = {}
        # loop through movie ids
        g_avg = total_stats[0] / total_stats[1]
        for id in tqdm(movieIds):
            # generate avrege weight for each movie id
            movie_dict[id] = (stats[id][0] + self.b * g_avg) / (stats[id][1] + self.b)
        self.movie_dict = movie_dict

    def predict(self, user_id):
        # from DataFrame column of movie ids to array
        return self.movie_dict
