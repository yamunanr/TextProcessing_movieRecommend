import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import pandas as pd
import numpy as np
from ast import literal_eval
import pandas_profiling as pf
from pandas_profiling import ProfileReport

credit_movies = pd.read_csv('./Data/tmdb_5000_credits.csv', sep=',')
movies = pd.read_csv('./Data/tmdb_5000_movies.csv')

credit_movies.columns = ['id', 'tittle', 'cast', 'crew']
data = credit_movies.merge(movies, on='id')


# print(data.columns)

# Demographic filtering : Weighted Rating (WR)
def weighted_avg(df):
    """
    :param df:
    :return:
    """
    weighted_rating = df['vote_average'].mean()
    print(weighted_rating)
    return weighted_rating

# credit_movies['cast'] = credit_movies['cast'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# print(credit_movies.columns)


# credit_movies_report = credit_movies.profile_report(title='EDA for Credit Data')
# credit_movies_report.to_file(output_file="credit_movies.html")
# print("\n View the report in credit_movies.html from your PC...\n ------------------------------------")

# movies_report = movies.profile_report(title='EDA for Movies Data')
# movies_report.to_file(output_file="movies.html")
# print("\n View the report in movies.html from your PC...\n ------------------------------------")
