from MovieData import MovieData
from RandomPredictor import RandomPredictor
from UserItemData import UserItemData


class Recommender:
    def __init__(self, predictor):
        self.x = {}
        self.predictor = predictor

    def fit(self, x):
        self.x = x
        self.predictor.fit(x)

    def recommend(self, user_id, n=10, rec_seen=True):
        predicted = self.predictor.predict(user_id)
        sorted_predicted = {k: v for k, v in sorted(predicted.items(), key=lambda item: item[1], reverse=True)}
        # print({k: v for k, v in sorted(predicted.items(), key=lambda item: item[1], reverse=False)})
        if rec_seen:
            rated_movies = self.x.data.loc[self.x.data['userID'] == user_id]['movieID'].to_numpy()
            for movie in rated_movies:
                del sorted_predicted[movie]
        print(len(sorted_predicted))
        return dict(list(sorted_predicted.items())[:n])


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    rp = RandomPredictor(1, 5)
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=5, rec_seen=False)
    for idmovie, val in rec_items.items():
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))
