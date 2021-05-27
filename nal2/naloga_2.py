"""
Instructions for implementing collaborative filtering
Predicting scores with similarity between products (+)
Write an ItemBasedPredictor class that accepts two parameters in the constructor: min_values and threshold (the default values of both should be 0). Based on the similarity between the products, the class should calculate the expected product rating for the selected user. Calculate the similarity with the corrected cosine distance. In class, implement the fit, predict, and similarity(self, p1, p2) methods, which returns the similarity between the p1 and p2 products. If the calculated similarity between the two products is less than the threshold or if we have less than min_values users who rated both movies, the similarity between the two products should be 0. I suggest calculating all similarities in the fit method and only using them in predict and fit.

Let’s just take movies that have at least 1000 ratings to make the calculation in the fit method faster. This implementation takes a few seconds.

md = MovieData('data/movies.dat')
uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
rp = ItemBasedPredictor()
rec = Recommender(rp)
rec.fit(uim)
print(uim.movies)
print("Similarity between the movies 'Men in black'(1580) and 'Ghostbusters'(2716): ", rp.similarity(1580, 2716))
print("Similarity between the movies 'Men in black'(1580) and 'Schindler's List'(527): ", rp.similarity(1580, 527))
print("Similarity between the movies 'Men in black'(1580) and 'Independence day'(780): ", rp.similarity(1580, 780))
Output:

Similarity between the movies 'Men in black'(1580) and 'Ghostbusters'(2716):  0.233955231768
Similarity between the movies 'Men in black'(1580) and 'Schindler's List'(527): 0.0 
Similarity between the movies 'Men in black'(1580) and 'Independence day'(780): 0.424661258447
If we use recommender for user 78, we get:

print("Predictions for 78: ")
rec_items = rec.recommend(78, n=15, rec_seen=False)
for idmovie, val in rec_items:
    print("Movie: {}, score: {}".format(md.get_title(idmovie), val))
Output:

Predictions for 78: 
Movie: Shichinin no samurai, score: 4.355734702553732
Movie: The Usual Suspects, score: 4.354681671325777
Movie: The Silence of the Lambs, score: 4.335305215729613
Movie: Sin City, score: 4.278687059261374
Movie: Monsters, Inc., score: 4.2175806874956665
Movie: The Incredibles, score: 4.207098405922537
Movie: The Lord of the Rings: The Fellowship of the Ring, score: 4.15279203714804
Movie: Batman Begins, score: 4.146413660330134
Movie: Die Hard, score: 4.125915501065452
Movie: Rain Man, score: 4.071535187988423
Movie: The Lord of the Rings: The Return of the King, score: 4.020237426179001
Movie: A Beautiful Mind, score: 4.015142457113093
Movie: Good Will Hunting, score: 4.009280780263334
Movie: The Lord of the Rings: The Two Towers, score: 3.9414763181545847
Movie: Indiana Jones and the Last Crusade, score: 3.7969765828982167
Most similar movies (+)
List 20 pairs of the most similar movies. If we again use only films that have at least 1000 ratings, we get:

Movie1: The Lord of the Rings: The Return of the King, Movie2: The Lord of the Rings: The Two Towers, similarity: 0.8439842148481411
Movie1: The Lord of the Rings: The Two Towers, Movie2: The Lord of the Rings: The Return of the King, similarity: 0.8439842148481411
Movie1: The Lord of the Rings: The Two Towers, Movie2: The Lord of the Rings: The Fellowship of the Ring, similarity: 0.8231885401761887
Movie1: The Lord of the Rings: The Fellowship of the Ring, Movie2: The Lord of the Rings: The Two Towers, similarity: 0.8231885401761887
Movie1: The Lord of the Rings: The Return of the King, Movie2: The Lord of the Rings: The Fellowship of the Ring, similarity: 0.8079374897442487
Movie1: The Lord of the Rings: The Fellowship of the Ring, Movie2: The Lord of the Rings: The Return of the King, similarity: 0.8079374897442487
Movie1: Kill Bill: Vol. 2, Movie2: Kill Bill: Vol. 2, similarity: 0.7372340224381033
Movie1: Kill Bill: Vol. 2, Movie2: Kill Bill: Vol. 2, similarity: 0.7372340224381033
Movie1: Star Wars: Episode V - The Empire Strikes Back, Movie2: Star Wars, similarity: 0.7021321132220316
Movie1: Star Wars, Movie2: Star Wars: Episode V - The Empire Strikes Back, similarity: 0.7021321132220316
Movie1: The Mask, Movie2: Ace Ventura: Pet Detective, similarity: 0.6616471778494041
Movie1: Ace Ventura: Pet Detective, Movie2: The Mask, similarity: 0.6616471778494041
Movie1: Star Wars: Episode VI - Return of the Jedi, Movie2: Star Wars: Episode V - The Empire Strikes Back, similarity: 0.5992253753778951
Movie1: Star Wars: Episode V - The Empire Strikes Back, Movie2: Star Wars: Episode VI - Return of the Jedi, similarity: 0.5992253753778951
Movie1: Star Wars: Episode I - The Phantom Menace, Movie2: Independence Day, similarity: 0.5610426219249982
Movie1: Independence Day, Movie2: Star Wars: Episode I - The Phantom Menace, similarity: 0.5610426219249982
Movie1: Austin Powers: The Spy Who Shagged Me, Movie2: Ace Ventura: Pet Detective, similarity: 0.5546511205201548
Movie1: Ace Ventura: Pet Detective, Movie2: Austin Powers: The Spy Who Shagged Me, similarity: 0.5546511205201548
Movie1: Star Wars: Episode VI - Return of the Jedi, Movie2: Star Wars, similarity: 0.5537849318137374
Movie1: Star Wars, Movie2: Star Wars: Episode VI - Return of the Jedi, similarity: 0.5537849318137374
Recommendation based on the currently viewed content (+)
What would you show in the category “Viewers who watched A also watched B”? Add a similarItems(self, item, n) method to the ItemBasedPredictor class, which returns n most similar movies to the selected movie.

rec_items = rp.similarItems(4993, 10)
print('Movies similar to "The Lord of the Rings: The Fellowship of the Ring": ')
for idmovie, val in rec_items:
    print("Movie: {}, score: {}".format(md.get_title(idmovie), val))
Output:

Movies similar to "The Lord of the Rings: The Fellowship of the Ring": 
Movie: The Lord of the Rings: The Two Towers, score: 0.8231885401761887
Movie: The Lord of the Rings: The Return of the King, score: 0.8079374897442487
Movie: Star Wars: Episode V - The Empire Strikes Back, score: 0.23961943073496453 F
Movie: Star Wars, score: 0.21965586527074088 
Movie: The Matrix, score: 0.2151555270688026 
Movie: Raiders of the Lost Ark, score: 0.19944276706345052 
Movie: The Usual Suspects, score: 0.18321188451910767 
Movie: Blade Runner, score: 0.16399681315410303 
Movie: Schindler's List, score: 0.16105905138148724 
Movie: Monty Python and the Holy Grail, score: 0.15780453798519137
Recommendation for yourself (+)
Make another recommendation for yourself using the "item-based" method; choose approx. 20 movies you know and rate them manually. Add your ratings to the movielens database and recommend 10 movies to yourself.

Prediction with Slope One method (+)
Write a SlopeOnePredictor that calculates the predicted value of ratings for products using the Slope one method. Example:

md = MovieData('data/movies.dat') 
uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000) 
rp = SlopeOnePredictor() 
rec = Recommender(rp) 
rec.fit(uim)

print("Predictions for 78: ") 
rec_items = rec.recommend(78, n=15, rec_seen=False) 
for idmovie, val in rec_items:
    print("Movie: {}, score: {}".format(md.get_title(idmovie), val))
Output:

Predictions for 78: 
Movie: The Usual Suspects, score: 4.325079182263173
Movie: The Lord of the Rings: The Fellowship of the Ring, score: 4.155293229840448
Movie: The Lord of the Rings: The Return of the King, score: 4.153135076202185
Movie: The Silence of the Lambs, score: 4.127978169643881
Movie: Shichinin no samurai, score: 4.119790444913598
Movie: The Lord of the Rings: The Two Towers, score: 4.083325894849594
Movie: Indiana Jones and the Last Crusade, score: 3.9670398355464194
Movie: The Incredibles, score: 3.9664496674557546
Movie: Good Will Hunting, score: 3.963362387354114
Movie: Sin City, score: 3.942619137615212
Movie: Batman Begins, score: 3.9375326640077017
Movie: A Beautiful Mind, score: 3.9140940935239508
Movie: Rain Man, score: 3.9107819079644943
Movie: Monsters, Inc., score: 3.8819375978658006
Movie: Finding Nemo, score: 3.8807711131654794
Hybrid predictor
Write a HybridPredictor class that composes a recommendation from the methods implemented so far. The hybrid predictor should therefore combine SlopeOne, ItemBased and “best rated” in one of the established ways.

Matrix factorization (*)
Write a MatrixFactorizationPredictor class that first factorizes the score matrix and then uses it to calculate scores. Matrix factorization is a popular method for recommendation systems, as it usually achieves better results than the nearest neighbor method; you can see more about it in (Y. Koren et al., Matrix factorization techniques for recommender systems OR G. Takacs et al., Scalable Collaborative Filtering Approaches for Large Recommender Systems).

Matrix factorization for implicit estimates (*)
The matrix factorization described above does not work best for implicit estimates such as e.g. buying an item or watching a movie. A good alternative is the probability model of matrix factorization; more about it in (Christopher C. Johnson; Logistic Matrix Factorization for Implicit Feedback Data; url: http://stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)

Recommender system with neural networks (*)
In many domains, neural networks outperform traditional methods and it appears to be the case with the recommender systems as well. Implement the neural network following the instructions from the blog https://towardsdatascience.com/paper-review-neural-collaborative-filtering-explanation-implementation-ea3e031b7f96.
"""


from imports.Data import UserItemData, MovieData
from imports.Recommender import Recommender
from Predictors import ItemBasedPredictor

def itemBasedPredictor():
    print("Runing item based predictor...")
    md = MovieData('../data/movies.dat')
    uim = UserItemData('../data/user_ratedmovies.dat', min_ratings=1000)
    print(uim.data)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    print(uim.movies)
    print("Similarity between the movies 'Men in black'(1580) and 'Ghostbusters'(2716): ", rp.similarity(1580, 2716))
    print("Similarity between the movies 'Men in black'(1580) and 'Schindler's List'(527): ", rp.similarity(1580, 527))
    print("Similarity between the movies 'Men in black'(1580) and 'Independence day'(780): ", rp.similarity(1580, 780))


if __name__ == "__main__":
    itemBasedPredictor()