"""
Evaluation of the recommender system
Method evaluate(self, test_data, n) (+)
Write a method of class Recommender that accepts test data test_data and calculates average MAE, RMSE, recall, accuracy, F1. For recall, accuracy, and F1, you’ll need to choose a few recommended products for each user. I decided to take the ones that the user rated better than their average. Note that you do not recommend already viewed products and that the parameter n indicates the number of recommended products.

Which of the methods you have already implemented works best?

Example:

md = MovieData('data/movies.dat')
uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000, end_date='1.1.2008')
rp = SlopeOnePredictor()
rec = Recommender(rp)
rec.fit(uim)

uim_test = UserItemData('data/user_ratedmovies.dat', min_ratings=200, start_date='2.1.2008')
mse, mae, precision, recall, f = rec.evaluate(uim_test, 20)
print(mse, mae, precision, recall, f)
Output:

0.756301097596 0.644772273689 0.10398146371932578 0.1433164697088829 0.12052066985624228
Incremental testing and cross-validation
Instead of a one-time test, it is better to repeat the distribution and assessment several times. In cross-validation, all scores are divided into a few parts (folds). E.g. suppose we divide the scores  into ten parts. Then repeat learning ten times on 9 folds and testing on 1 fold (each time different fold remains for testing). The final values of the statistics are the average values across these individual tests.

Incremental testing is the best approximation of the operation of a real system when event dates (e.g. for scores) are available. We first choose a start date and only learn from the scores up to that date. We test this system on a window, e.g. scores submitted the following week. We then incorporate these assessments into the learning set and test on the week that follows. We repeat this until we run out of scores. In the end, we average the results.
"""