import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from pandas.core.frame import DataFrame
from pandas.io.formats.format import SeriesFormatter
from scipy import spatial
from tqdm import tqdm

class ItemBasedPredictor:
    def __init__(self, min_values=0, treshold=0):
        self.min_values     = min_values
        self.treshold       = treshold

    def fit(self, user_data):
        # save data of ratings
        self.data = user_data.data
        # save the pivot table (userID:movieID => value in tabe is the rating of 
        # the movie by the user | or NaN is no rating has been given)
        self.rating_table = pd.pivot_table(self.data, values="rating", index=["userID"], columns=['movieID'])
        self.rating_table_with_nan = pd.pivot_table(self.data, values="rating", index=["userID"], columns=['movieID'])
        # adjust ratings in the table (substract avg user rating for every user)
        # get all uniqe user ids
        user_ids = self.data['userID'].unique()
        # loop through user 
        for index, id in enumerate(user_ids):
            # get ratings of the user for all movies (may include NaN for not rating a movie)
            user = self.rating_table.iloc[index]
            # remove NaN entries and calulcate mean of non-NaN entries
            avg_rating = user[~np.isnan(user)].mean()
            # replace NaN values with the avg rating
            self.rating_table.iloc[index] = self.rating_table.iloc[index].fillna(avg_rating)
            # substract acg from all ratings (NaN and avg is now at 0) and 
            self.rating_table.iloc[index] = self.rating_table.iloc[index] - avg_rating
        # dataframe where both axis are movieID. The value of the field is currently 
        # NaN but will be the similiarity between movies (by user ratings)
        self.adj_mat = self.rating_table.T @ self.rating_table
        # loop thorugh every row (movie)
        for index_row, arr_value in tqdm(enumerate(self.adj_mat.values), total=len(self.adj_mat.values)):
            # for every row go though every column (column is also a movieID)
            for index_col, value in enumerate(arr_value):
                # find movie ids of row and column (currently index_row and index_column 
                # are indexes in array, not actual IDs)
                id_row = self.rating_table.columns[index_row]
                id_col = self.rating_table.columns[index_col]
                # if they are not the same (not the diagonal) we compare the movies by 
                # rating and save the similarity score
                if id_row != id_col:
                    self.adj_mat.at[id_row, id_col] = self.similarity(id_row, id_col)
                else:
                    self.adj_mat.at[id_row, id_col] = 0

        
    def predict(self, user_id):
        user_ratings = self.rating_table_with_nan.iloc[user_id].dropna()
        # ids of all movies that the user rated
        user_rated_movies = user_ratings.keys().to_numpy()
        # ids of all movies with ratings
        all_movie_ids = sorted(self.rating_table_with_nan.columns)
        # dict to store the generated ratings or the user (will be returened at the end)
        movie_ratings = {}
        for index, id in enumerate(all_movie_ids):
            similarities_of_the_movie = self.adj_mat[id]
            similarities_to_user_rated_movies = similarities_of_the_movie[user_rated_movies]
            max_similarity = similarities_to_user_rated_movies.max()
            index_max_similarity = similarities_to_user_rated_movies.idxmax()
            movie_ratings[id] = max_similarity * user_ratings[index_max_similarity]

        return movie_ratings

    def similarity(self, p1, p2):
        # get movie 1 and movie 2 ratings
        p1_ratings = self.rating_table[p1]
        p2_ratings = self.rating_table[p2]
        # create temporary table of the ratings
        # if not enough ratings return simmilarity as 0
        if len(p1_ratings) < self.min_values:
            return 0
        # calc similarity
        similarity = 1 - spatial.distance.cosine(p1_ratings, p2_ratings)
        # if not similar enough, return 0
        if similarity < self.treshold:
            return 0
        # return similarity
        return similarity
        
        


class MovieSimilarityBasedPredictor:
    def __init__(self, min_values=0, treshold=0):
        self.min_values     = min_values
        self.treshold       = treshold

    def fit(self, user_data):
        # save data of ratings
        self.data = user_data.data
        # save the pivot table (userID:movieID => value in tabe is the rating of 
        # the movie by the user | or NaN is no rating has been given)
        self.rating_table = pd.pivot_table(self.data, values="rating", index=["userID"], columns=['movieID'])
        self.rating_table_with_nan = pd.pivot_table(self.data, values="rating", index=["userID"], columns=['movieID'])
        # adjust ratings in the table (substract avg user rating for every user)
        # get all uniqe user ids
        user_ids = self.data['userID'].unique()
        # loop through user 
        for index, id in enumerate(user_ids):
            # get ratings of the user for all movies (may include NaN for not rating a movie)
            user = self.rating_table.iloc[index]
            # remove NaN entries and calulcate mean of non-NaN entries
            avg_rating = user[~np.isnan(user)].mean()
            # replace NaN values with the avg rating
            self.rating_table.iloc[index] = self.rating_table.iloc[index].fillna(avg_rating)
            # substract acg from all ratings (NaN and avg is now at 0) and 
            self.rating_table.iloc[index] = self.rating_table.iloc[index] - avg_rating
        # dataframe where both axis are movieID. The value of the field is currently 
        # NaN but will be the similiarity between movies (by user ratings)
        self.adj_mat = self.rating_table.T @ self.rating_table
        # loop thorugh every row (movie)
        for index_row, arr_value in tqdm(enumerate(self.adj_mat.values), total=len(self.adj_mat.values)):
            # for every row go though every column (column is also a movieID)
            for index_col, value in enumerate(arr_value):
                # find movie ids of row and column (currently index_row and index_column 
                # are indexes in array, not actual IDs)
                id_row = self.rating_table.columns[index_row]
                id_col = self.rating_table.columns[index_col]
                # if they are not the same (not the diagonal) we compare the movies by 
                # rating and save the similarity score
                if id_row != id_col:
                    self.adj_mat.at[id_row, id_col] = self.similarity(id_row, id_col)
                else:
                    self.adj_mat.at[id_row, id_col] = 0

        
    def predict(self, user_id):
        # all movie ids
        all_movie_ids = sorted(self.rating_table_with_nan.columns)
        # dict to store the generated ratings or the user (will be returened at the end)
        movie_ratings = {}
        for movie_id in tqdm(all_movie_ids):
            # all similarities with other movies
            similarities_of_the_movie = self.adj_mat[movie_id]
            # biggest similatires
            max_similarity = similarities_of_the_movie.max()
            # id of the most similar movie
            similar_movie_id = similarities_of_the_movie.idxmax()
            # save the movie with similarity for output
            movie_ratings[movie_id] = [similar_movie_id, max_similarity]
        return movie_ratings

    def similarity(self, p1, p2):
        # get movie 1 and movie 2 ratings
        p1_ratings = self.rating_table[p1]
        p2_ratings = self.rating_table[p2]
        # create temporary table of the ratings
        # if not enough ratings return simmilarity as 0
        if len(p1_ratings) < self.min_values:
            return 0
        # calc similarity
        similarity = 1 - spatial.distance.cosine(p1_ratings, p2_ratings)
        # if not similar enough, return 0
        if similarity < self.treshold:
            return 0
        # return similarity
        return similarity
