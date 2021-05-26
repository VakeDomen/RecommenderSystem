from collections import OrderedDict

import pandas as pd

from MovieData import MovieData
from Recommender import Recommender
from UserItemData import UserItemData


class AveragePredictor:

    def __init__(self, b):
        self.b = b
        self.data = {}

    def calculate_average(self, vs, n, g_avg):
        return (vs + self.b * g_avg) / (n + self.b)

    def fit(self, x):
        self.data = pd.DataFrame(x.data)

        g_avg = self.data['rating'].mean()

        self.data = self.data.groupby('movieID').agg(vs=('rating', sum), n=('rating', 'count'))
        self.data['rating'] = self.data.apply(lambda row: self.calculate_average(row['vs'], row['n'], g_avg), axis=1)

    def predict(self, x):
        print(self.data['rating'])
        dict = self.data['rating'].to_dict()
        print(dict)
        print(OrderedDict(sorted(dict.items(), key=lambda kv: kv[1]['rating'])))
#     todo: figure out how to sort dictionary


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    ap = AveragePredictor(b=0)
    ap.fit(uim)
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=5, rec_seen=False)
    for idmovie, val in rec_items.items():
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))
