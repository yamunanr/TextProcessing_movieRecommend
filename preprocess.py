import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
import pandas as pd
import numpy as np
from ast import literal_eval
import pandas_profiling as pf
from pandas_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

credit_movies = pd.read_csv('./Data/tmdb_5000_credits.csv', sep=',')
movies = pd.read_csv('./Data/tmdb_5000_movies.csv')

credit_movies.columns = ['id', 'tittle', 'cast', 'crew']
data = credit_movies.merge(movies, on='id')


# print(data.columns)

# preprocessing methods:
# function, weighted_rating() and define a new feature score, of which we'll calculate the value by applying this function to our DataFrame of qualified movies:


# Demographic filtering : Weighted Rating (WR)
def filtering(df):
    """
    :param df: data
    :return: filtered movies
    """
    weighted_avg_rating = df['vote_average'].mean()
    print("Mean rating for all movies::", weighted_avg_rating)
    minimum_votes = df['vote_count'].quantile(
        0.9)  # Using 90th percentile, movie votes to be equal to or more than 90% as cutoff --> feature in charts
    print("min_votes_movies:", minimum_votes)
    # filter movies
    q_movies = df.copy().loc[df['vote_count'] >= minimum_votes]
    print('Filtered Movies: \n ---------------------------\n', q_movies)

    def weighted_rating(x=df, m=minimum_votes, C=weighted_avg_rating):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v / (v + m) * R) + (m / (m + v) * C)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    # Sort movies based on score calculated above
    q_movies = q_movies.sort_values('score', ascending=False)
    # Print the top 15 movies
    q_movies = q_movies[['title', 'vote_count', 'vote_average', 'score']]
    print(q_movies)
    return q_movies


filtering(data)


def popular_movies(df):
    """

    """
    pm = df.sort_values('popularity', ascending=False).head(15)
    fig = px.bar(pm, x="title", y=["popularity"], title="Top 15 Popular Movies")
    fig.show()


# popular_movies(data)

# Text-preprocessing
# def text_processing(df):
print("Data used for recommender processing\n", data['overview'].head(5))
# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
# Replace NaN with an empty string
data['overview'] = data['overview'].fillna('')
# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(data['overview'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(data.index, index=data['title']).drop_duplicates()


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    title = input("Enter Movie Name:")

    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    print("10 Most Similar Movies to Recommend", data['title'].iloc[movie_indices])
    return data['title'].iloc[movie_indices]


get_recommendations()

