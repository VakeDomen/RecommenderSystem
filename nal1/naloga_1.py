from Predictors import RandomPredictor, ViewsPredictor, AveragePredictor
from Recommender import Recommender
from Data import UserItemData, MovieData

def recommendation():
    print("\nRunning recommendations.....")
    md = MovieData('../data/movies.dat') 
    uim = UserItemData('../data/user_ratedmovies.dat') 
    rp = RandomPredictor(1, 5) 
    rec = Recommender(rp) 
    rec.fit(uim) 
    rec_items = rec.recommend(78, n=5, rec_seen=False) 
    for idmovie, val in rec_items: 
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def readRatings():
    print("\nReading ratings....")
    uim = UserItemData('../data/user_ratedmovies.dat')
    print(uim.nratings())
    uim = UserItemData('../data/user_ratedmovies.dat', from_date='12.1.2007', to_date='16.2.2008', min_ratings=100)
    print(uim.nratings())

def readMovies():
    print("\nReading movies....")
    md = MovieData('../data/movies.dat')
    print(md.get_title(65091))

def randomPredictor():
    print("\nRunning random predictions....")
    md = MovieData('../data/movies.dat')
    uim = UserItemData('../data/user_ratedmovies.dat') 
    rp = RandomPredictor(1, 5) 
    rp.fit(uim) 
    pred = rp.predict(78) 
    print(type(pred)) 
    items = [1, 3, 20, 50, 100] 
    for item in items: 
        print("Movie: {}, score: {}".format(md.get_title(item), pred[item]))

def averagePredictor():
    print("\nRunning avrage predictions....")
    md = MovieData('../data/movies.dat') 
    uim = UserItemData('../data/user_ratedmovies.dat') 
    ap = AveragePredictor(0) 
    rec = Recommender(ap) 
    rec.fit(uim) 
    rec_items = rec.recommend(78, n=5, rec_seen=False) 
    for idmovie, val in rec_items: 
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

def viewsPredictor():
    print("\nRunning views predictions...")
    md = MovieData('../data/movies.dat') 
    uim = UserItemData('../data/user_ratedmovies.dat') 
    ap = ViewsPredictor() 
    rec = Recommender(ap) 
    rec.fit(uim) 
    rec_items = rec.recommend(78, n=5, rec_seen=True) 
    for idmovie, val in rec_items: 
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

# ---------------- MAIN ----------------

if __name__ == "__main__":
    readRatings()
    readMovies()
    randomPredictor()
    recommendation()
    averagePredictor()
    viewsPredictor()