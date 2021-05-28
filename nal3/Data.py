import random
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import math

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

    def getTestSetData(self, user_id, test_size_percentage=0.2):
        # all user ratings
        # print(self.data.groupby('userID').agg(count=('rating', 'count')).idxmax())
        
        user_data = self.data.loc[self.data['userID'] == user_id]
        # size of the test set
        test_set_size = math.floor(len(user_data) * test_size_percentage)
        # save the test set from user ratings
        test_set = user_data[:test_set_size]
        # remove test set from all user ratings to generate training set
        trainig_set = self.data.drop(self.data.index[test_set.index.to_numpy()])
        # overide data with training set (data w/o test set)
        self.data = trainig_set

        return test_set


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
