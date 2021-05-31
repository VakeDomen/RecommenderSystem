from imports.Data import UserItemData, MovieData
from imports.Recommender import Recommender, RecommenderTopSimilar
from Predictors import ItemBasedPredictor, MovieSimilarityBasedPredictor, SlopeOnePredictor

def itemBasedPredictor():
    print("\nRuning item based predictor...")
    md = MovieData('../data/movies.dat')
    uim = UserItemData('../data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    # print(uim.movies)
    print("Similarity between the movies 'Men in black'(1580) and 'Ghostbusters'(2716): ", rp.similarity(1580, 2716))
    print("Similarity between the movies 'Men in black'(1580) and 'Schindler's List'(527): ", rp.similarity(1580, 527))
    print("Similarity between the movies 'Men in black'(1580) and 'Independence day'(780): ", rp.similarity(1580, 780))

    print("\nPredictions for 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def movieSimilarityPrediction():
    print("\nRuning movie similarity prediction...")
    md = MovieData('../data/movies.dat')
    uim = UserItemData('../data/user_ratedmovies.dat', min_ratings=1000)
    rp = MovieSimilarityBasedPredictor()
    rec = RecommenderTopSimilar(rp)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=20, rec_seen=False)
    # print(rec_items)
    for idmovie, val in rec_items:
        # print(val)
        print("Movie1: {}, Movie2: {}, similarity: {}".format(md.get_title(idmovie), md.get_title(val[0]), val[1]))

def mostSimilarToMovie():
    print("\nRunning most similar movies...")
    md = MovieData('../data/movies.dat')
    uim = UserItemData('../data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rp.similarItems(4993, 5)
    print('Movies similar to "The Lord of the Rings: The Fellowship of the Ring": ')
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def listMoviesforMe():
    print("\nRuning my predictions...")
    md = MovieData('../data/movies.dat')
    uim = UserItemData('../data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    print("Predictions for myself (user 666): ")
    rec_items = rec.recommend(666, n=10, rec_seen=False)
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def slopeOnePredictor():
    print("\nRunning slope one predictor....")
    md = MovieData('../data/movies.dat') 
    uim = UserItemData('../data/user_ratedmovies.dat', min_ratings=1000) 
    rp = SlopeOnePredictor() 
    rec = Recommender(rp) 
    rec.fit(uim)

    print("\nPredictions for 78: ") 
    rec_items = rec.recommend(78, n=15, rec_seen=False) 
    for idmovie, val in rec_items:
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

if __name__ == "__main__":
    itemBasedPredictor()
    movieSimilarityPrediction()
    mostSimilarToMovie()
    listMoviesforMe()
    slopeOnePredictor()