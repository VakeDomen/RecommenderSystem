import pandas as pd

from MovieData import MovieData
from Recommender import Recommender
from UserItemData import UserItemData


class ViewsPredictor:
    def __init__(self):
        self.data = {}

    def fit(self, x):
        self.data = pd.DataFrame(x.data)
        self.data = self.data.groupby('movieID').agg(count=('rating', 'count'))

    def predict(self, x):
        return self.data['count'].to_dict()


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    vp = ViewsPredictor()
    vp.fit(uim)
    rec = Recommender(vp)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=5, rec_seen=True)
    for idmovie, val in rec_items.items():
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))
