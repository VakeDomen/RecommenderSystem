import random

import pandas as pd

from UserItemData import UserItemData
from MovieData import MovieData
import numpy as np


class RandomPredictor:
    def __init__(self, min_rating, max_rating):
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.data = {}

    def fit(self, x):
        self.data = x.data

    def predict(self, user_id):
        unique_movies = self.data[['movieID', 'rating']].drop_duplicates('movieID', keep='first')
        random_ratings = np.random.randint(self.min_rating, self.max_rating + 1, len(unique_movies))
        unique_movies['rating'] = random_ratings
        return dict(zip(unique_movies['movieID'], unique_movies['rating']))


if __name__ == "__main__":
    md = MovieData('../data/movies.dat')
    uim = UserItemData('../data/user_ratedmovies.dat')
    rp = RandomPredictor(1, 5)
    rp.fit(uim)
    pred = rp.predict(78)
    print(type(pred))
    items = [1, 3, 20, 50, 100]
    for item in items:
        print("Movie: {}, score: {}".format(md.get_title(item), pred[item]))
