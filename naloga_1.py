"""
Basic recommendations
The data
We will use the derived movielens data in these instructions. Unlike the original data set, this one contains much more than just estimates. The data description is in the README file. You can use any large enough data for your seminar.

Reading ratings (+)
Write a class (e.g. named UserItemData) in which you will store the read data. I suggest that you have a path argument and three optional arguments in the class constructor: from_date (from which date to read data), to_date (by which date to read data), and min_ratings (minimum number of ratings a movie should have). Add a method to the class that tells you how many ratings it has read. I suggest that you add a method that saves the read data or read the data using the pickle library. 

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