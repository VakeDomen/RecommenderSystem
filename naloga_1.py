"""
Basic recommendations
The data
We will use the derived movielens data in these instructions. Unlike the original data set, this one contains much more than just estimates. The data description is in the README file. You can use any large enough data for your seminar.

Reading ratings (+)
Write a class (e.g. named UserItemData) in which you will store the read data. I suggest 
that you have a path argument and three optional arguments in the class constructor: from_date 
(from which date to read data), to_date (by which date to read data), and min_ratings (minimum 
number of ratings a movie should have). Add a method to the class that tells you how many ratings 
it has read. I suggest that you add a method that saves the read data or read the data using 
the pickle library. 

Example:

uim = UserItemData('data/user_ratedmovies.dat')
print(uim.nratings())

uim = UserItemData('data/user_ratedmovies.dat', start_date = '12.1.2007', end_date='16.2.2008', min_ratings=100)
print(uim.nratings())
Output:

855598
72784
Reading movies (+)
Write a class that reads the movie file and has a get_title(movieID) method that returns its title for a given movie ID. Example:

md = MovieData('data/movies.dat')
print(md.get_title(1))
Output:

Toy story
Predictor
The word "predictor" will be used to denote classes that, in a certain way, assess for a certain user the rating he would give to the movies or the products available to him. These classes will have a fit(self, X) method, where X is of type UserItemData, and a predict(self, user_id) method, where user_id is the user ID. We will use the fit method to learn the model, and the predict to calculate the recommended values for a given user. Recommended values can be e.g. returned as a dictionary.

Random predictor (+)
Write a RandomPredictor class that accepts a minimum and maximum rating in the constructor, and the predict method returns a random value (between min and max) for each product.

md = MovieData('data/movies.dat')
uim = UserItemData('data/user_ratedmovies.dat') 
rp = RandomPredictor(1, 5) rp.fit(uim) 
pred = rp.predict(78) 
print(type(pred)) 
items = [1, 3, 20, 50, 100] 
for item in items: 
    print("Movie: {}, score: {}".format(md.get_title(item), pred[item]))
Output:

<class 'dict'>
Movie: Toy story, score: 5
Movie: Grumpy Old Men, score: 4
Movie: Money Train, score: 5
Movie: The Usual Suspects, score: 5
Movie: City Hall, score: 3
Recommendation (+)
Write a class Recommender. The class should accept the predictor in the constructor and have two methods: fit(self, X) and recommend(self, userID, n = 10, rec_seen = True). The first method has the same function as in the predictor, while the second method returns an edited list of recommended products for the userID user. The parameter n determines the number of recommended movies, and with rec_seen we determine whether we want the already watched movies (those to which the user has already given a rating) or not.

md = MovieData('data/movies.dat') 
uim = UserItemData('data/user_ratedmovies.dat') 
rp = RandomPredictor(1, 5) 
rec = Recommender(rp) 
rec.fit(uim) 
rec_items = rec.recommend(78, n=5, rec_seen=False) 
for idmovie, val in rec_items: 
    print("Movie: {}, score: {}".format(md.get_title(idmovie), val)) 
Output:

Movie: Toy story, score: 5 
Movie: Sudden Death, score: 5 
Movie: Dracula: Dead and Loving It, score: 5 
Movie: Money Train, score: 5 
Movie: It Takes Two, score: 5
Average predictor (+)
Write the class AveragePredictor, which takes the parameter b in the constructor, where b >= 0. In the fit method, calculate the average for each movie according to the formula avg = (vs + b * g_avg) / (n + b), where:

vs is the sum of all ratings for this movie,
n is the number of scores this movie received,
g_avg is the average of all movies,
b is the parameter of the formula for the average. If b=0, it is a simple average.
If we use AveragePredictor(b=0) in the recommender and the rest is the same as in the previous section, we get:

Movie: Sonnenallee, score: 5.0 
Movie: Vals Im Bashir, score: 5.0 
Movie: Britannia Hospital, score: 5.0 
Movie: Il mio viaggio in Italia, score: 5.0 
Movie: Shu dan long wei, score: 5.0
However, if we use AveragePredictor(b=100), the values change:

Movie: The Usual Suspects, score: 4.225944245560473 
Movie: The Godfather: Part II, score: 4.146907937910189 
Movie: Cidade de Deus, score: 4.116538340205236 
Movie: The Dark Knight, score: 4.10413904093503 
Movie: 12 Angry Men, score: 4.103639627096175
Recommending the most watched movies (+)
Write a ViewsPredictor that returns the number of views for each movie. This is the recommendation of most viewed movies. For our example, you will get the following:

Movie: The Lord of the Rings: The Fellowship of the Ring, score: 1576 
Movie: The Lord of the Rings: The Two Towers, score: 1528 
Movie: The Lord of the Rings: The Return of the King, score: 1457 
Movie: The Silence of the Lambs, score: 1431 
Movie: Shrek, score: 1404
Recommendation of controversial films
How would you rate controversy (products that have a lot of good and a lot of bad reviews)? Is the “non-personalized” way of recommending suitable for such products? Write a predictor for the most controversial products where a film can be controversial if it has at least n ratings. Use the standard deviation of the estimates for the measure. If a film must have at least 100 ratings, we get:

md = MovieData('data/movies.dat') 
uim = UserItemData('data/user_ratedmovies.dat') 
rp = STDPredictor(100) 
rec = Recommender(rp) rec.fit(uim) 
rec_items = rec.recommend(78, n=5, rec_seen=False) 
for idmovie, val in rec_items: 
    print("Movie: {}, score: {}".format(md.get_title(idmovie), val))
Output:

Movie: Plan 9 from Outer Space, score: 1.3449520951495717 
Movie: The Passion of the Christ, score: 1.281493459525735 
Movie: The Texas Chainsaw Massacre, score: 1.235349321908819 
Movie: Jackass Number Two, score: 1.2189769976366684 
Movie: White Chicks, score: 1.1899581424297319

"""


import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime



class UserItemData:
    def __init__(self, path, from_date="1.1.1900", to_date="31.12.2025", min_ratings=0):
        # save data
        self.path           = path
        self.from_date      = from_date
        self.to_date        = to_date
        self.min_ratings    = min_ratings

        # import dataset
        from_data   = self.from_date.split('.')
        to_data     = self.to_date.split('.')
        self.data = pd.read_csv(self.path, sep="\t")
        # rename cols so to_datetime can recognize them
        self.data.columns = ['userID', 'movieID', 'rating', 'day', 'month', 'year', 'date_hour', 'date_minute', 'date_second']
        # add column timestamp and format to datetime
        self.data['timestamp'] = pd.to_datetime(self.data[['day', 'month', 'year']])
        # filter out smaller by date
        self.data = self.data[
            self.data['timestamp'] >= datetime.strptime(self.from_date, '%d.%m.%Y')
        ]
        # filter out bigger by date
        self.data = self.data[
            self.data['timestamp'] <= datetime.strptime(self.to_date, '%d.%m.%Y')
        ]

        # filtr by minimum rating
        self.data = self.data.groupby('movieID').filter(
            lambda group: len(group) > self.min_ratings
        )
        



    def nratings(self):
        return len(self.data)

class MovieData:
    def __init__(self, path):
        # save data
        self.path = path
        self.data = pd.read_csv(self.path, sep="\t")
        
    def get_title(self, id):
        # filter by id
        moviesById = self.data.query(f'id=={id}')
        # extact title column (from array of site 1 / 0)
        titles = moviesById['title']
        # return first element (title)
        return titles.iloc[0]

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


def readRatings():
    uim = UserItemData('data/user_ratedmovies.dat')
    print(uim.nratings())
    uim = UserItemData('data/user_ratedmovies.dat', from_date='12.1.2007', to_date='16.2.2008', min_ratings=100)
    print(uim.nratings())

def readMovies():
    md = MovieData('data/movies.dat')
    print(md.get_title(65091))

def randomPredictor():
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat') 
    rp = RandomPredictor(1, 5) 
    rp.fit(uim) 
    pred = rp.predict(78) 
    print(type(pred)) 
    items = [1, 3, 20, 50, 100] 
    for item in items: 
        print("Movie: {}, score: {}".format(md.get_title(item), pred[item]))

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

def recommendation():
    md = MovieData('data/movies.dat') 
    uim = UserItemData('data/user_ratedmovies.dat') 
    rp = RandomPredictor(1, 5) 
    rec = Recommender(rp) 
    rec.fit(uim) 
    rec_items = rec.recommend(78, n=5, rec_seen=False) 
    for idmovie, val in rec_items: 
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))


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



def averagePredictor():
    md = MovieData('data/movies.dat') 
    uim = UserItemData('data/user_ratedmovies.dat') 
    ap = AveragePredictor(0) 
    rec = Recommender(ap) 
    rec.fit(uim) 
    rec_items = rec.recommend(78, n=5, rec_seen=False) 
    for idmovie, val in rec_items: 
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))



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


def viewsPredictor():
    md = MovieData('data/movies.dat') 
    uim = UserItemData('data/user_ratedmovies.dat') 
    ap = ViewsPredictor() 
    rec = Recommender(ap) 
    rec.fit(uim) 
    rec_items = rec.recommend(78, n=5, rec_seen=True) 
    for idmovie, val in rec_items: 
        print("Movie: {}, score: {}".format(md.get_title(idmovie), val))

if __name__ == "__main__":
    readRatings()
    readMovies()
    randomPredictor()
    recommendation()
    averagePredictor()
    viewsPredictor()