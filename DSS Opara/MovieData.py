import pandas as pd
from pandas import DataFrame


class MovieData(object):

    def __init__(self, path):
        self.path = path
        self.file = pd.read_csv(self.path, sep="\t")
        self.data = DataFrame(self.file, columns=['id', 'title'])

    def get_title(self, movie_id):
        return self.data.loc[self.data['id'] == movie_id].iat[0, 1]


if __name__ == "__main__":
    md = MovieData('data/movies.dat')
    print(md.get_title(1))
