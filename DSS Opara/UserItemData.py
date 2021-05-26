from datetime import datetime

import pandas as pd
from pandas import DataFrame


class UserItemData(object):

    def __init__(self, path, start_date=None, end_date=None, min_ratings=None):
        self.path = path
        self.start_date = start_date
        self.end_date = end_date
        self.min_ratings = min_ratings
        self.file = pd.read_csv(self.path, sep="\t")

        self.data = DataFrame(self.file, columns=['userID', 'movieID', 'rating', 'date_day', 'date_month', 'date_year'])
        self.data.columns = ['userID', 'movieID', 'rating', 'day', 'month', 'year']
        self.data['date'] = pd.to_datetime(self.data[['day', 'month', 'year']])
        self.data = self.data[['userID', 'movieID', 'rating', 'date']]

        if self.start_date is not None:
            self.data = self.data[self.data['date'] >= datetime.strptime(self.start_date, '%d.%m.%Y')]
        if self.end_date is not None:
            self.data = self.data[self.data['date'] < datetime.strptime(self.end_date, '%d.%m.%Y')]
        if min_ratings is not None:
            self.data = self.data.groupby('movieID').filter(lambda group: len(group) > self.min_ratings)

    def nratings(self):
        return len(self.data)


if __name__ == "__main__":
    uim = UserItemData('data/user_ratedmovies.dat')
    print(uim.nratings())

    uim = UserItemData('data/user_ratedmovies.dat', start_date='12.1.2007', end_date='16.2.2008', min_ratings=100)
    print(uim.nratings())
