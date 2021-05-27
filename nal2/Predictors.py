import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from pandas.core.frame import DataFrame
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
        user_rated_movies = self.rating_table.query(f'userID=={user_id}').squeeze().dropna()

        print(user_rated_movies)
        return

    def similarity(self, p1, p2):
        p1_ratings = self.rating_table[p1]
        p2_ratings = self.rating_table[p2]
        tmp_table = DataFrame()
        tmp_table[p1] = p1_ratings
        tmp_table[p2] = p2_ratings
        tmp_table = tmp_table.dropna()

        if len(tmp_table) < self.min_values:
            return 0

        p1_ratings = tmp_table[p1].to_numpy()
        p2_ratings = tmp_table[p2].to_numpy()
        similarity = 1 - spatial.distance.cosine(p1_ratings, p2_ratings)
        # similiraty = 1 - dot(p1_ratings, p2_ratings) / (norm(p1_ratings) * norm(p2_ratings))
        if similarity < self.treshold:
            return 0

        return similarity
        
        
