from Data import UserItemData, MovieData
from imports.Recommender import Recommender
from imports.Predictors import AveragePredictor, RandomPredictor, SlopeOnePredictor, ItemBasedPredictor, ViewsPredictor


if __name__ == "__main__":
    print("Running evaluator...")
    md = MovieData('../data/movies.dat')
    uid = UserItemData('../data/user_ratedmovies.dat', min_ratings=1000)
    test_set = uid.getTestSetData(2053)
    rp = SlopeOnePredictor()
    rec = Recommender(rp)
    rec.fit(uid)
    
    mae, rmse, precision, recall, f1 = rec.evaluate(2053, test_set)
    print("------- Evaluation results: -------")
    print('MAE:   \t\t' + str(mae))
    print('RMSE:   \t' + str(rmse))
    print('precision: \t' + str(precision))
    print('recall: \t' + str(recall))
    print('f1:    \t\t' + str(f1))

