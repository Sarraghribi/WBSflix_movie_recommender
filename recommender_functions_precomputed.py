import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests



## Popularity ranking

def popularity_ranking(n, sorted_popularity_df, exclude_movies=None):
    '''Main function for running popularity ranking.'''
    
    if exclude_movies == None:
        exclude_movies = []
    
    # Exclude movies that have already been recommended
    recommendations = sorted_popularity_df.loc[~sorted_popularity_df["movieId"].isin(exclude_movies)]
    
    top_n_movie_ids = recommendations.head(n)["movieId"].to_list()
    top_n_movie_titles = recommendations.head(n)["title"].to_list()
    
    return top_n_movie_ids, top_n_movie_titles



## Item-based collaborative filtering

def item_based_cf(ref_mov, n, mov_corrs, mov_df, exclude_movies=None):
    '''Main function for running item-based collaborative filtering.'''
    
    if exclude_movies == None:
        exclude_movies = []
        
    # Get correlations of ref_movie ratings with all other movie ratings
    ref_mov_corr = pd.DataFrame(mov_corrs[ref_mov].dropna().sort_values(ascending=False))
    
    # Assemble list of movies, sorted by predicted rating
    recommendations = ref_mov_corr.merge(mov_df, left_index=True, right_on="movieId", how="left")
    
    # Exclude movies that have already been recommended
    recommendations_excl = recommendations.loc[~recommendations["movieId"].isin(exclude_movies)]
    
    # Get Top n
    top_n_movie_ids = recommendations_excl.head(n)["movieId"].to_list()
    top_n_movie_titles = recommendations_excl.head(n)["title"].to_list()
    
    return top_n_movie_ids, top_n_movie_titles

def item_based_cf_old(ref_mov, n, rat_df, mov_df, exclude_movies=None):
    '''Main function for running item-based collaborative filtering.'''
    
    if exclude_movies == None:
        exclude_movies = []
        
    # Create sparse matrix
    rating_crosstab = pd.pivot_table(data=rat_df, values="rating", index="userId", columns="movieId")
    ref_mov_rating = rating_crosstab.pop(ref_mov)
    
    # Only consider movie pairs where each movie was seen by at least m of the same users
    m = 5
    check_rating_nums = rating_crosstab.multiply(ref_mov_rating.values, axis="index").count()
    drop_movies = check_rating_nums[check_rating_nums < m].index.to_list()
    rating_crosstab.drop(columns=drop_movies, inplace=True)
    
    # Get correlations of ref_movie ratings with all other movie ratings
    mov_corr = rating_crosstab.corrwith(ref_mov_rating)
    mov_corr = pd.DataFrame(mov_corr.dropna().sort_values(ascending=False), columns=["Pearson_r"])
    
    # Assemble list of movies, sorted by predicted rating
    recommendations = mov_corr.merge(mov_df, left_index=True, right_on="movieId", how="left")
    
    # Exclude movies that have already been recommended
    recommendations_excl = recommendations.loc[~recommendations["movieId"].isin(exclude_movies)]
    
    # Get Top n
    top_n_movie_ids = recommendations_excl.head(n)["movieId"].to_list()
    top_n_movie_titles = recommendations_excl.head(n)["title"].to_list()
    
    return top_n_movie_ids, top_n_movie_titles


## User-based collaborative filtering

def user_based_cf(ref_user, n, predicted_ratings, mov_df, exclude_movies=None):
    
    '''Main function for running user-based collaborative filtering.'''
    
    if exclude_movies == None:
        exclude_movies = []
    
    user_pred_ratings = predicted_ratings.loc[[ref_user]].T.dropna()
    user_pred_ratings = user_pred_ratings.rename(columns={ref_user:"predicted_rating"})

    # Assemble list of movies, sorted by predicted rating
    recommendations = user_pred_ratings.merge(mov_df, left_index=True, right_on="movieId", how="left")
    recommendations = recommendations.sort_values(by="predicted_rating", ascending=False)

    # Exclude movies that have already been recommended
    recommendations_excl = recommendations.loc[~recommendations["movieId"].isin(exclude_movies)]

    # Get Top n
    top_n_movie_ids = recommendations_excl.head(n)["movieId"].to_list()
    top_n_movie_titles = recommendations_excl.head(n)["title"].to_list()

    return top_n_movie_ids, top_n_movie_titles




### Get movie images from TMDB API

tmdb_endpoint = "https://api.themoviedb.org/3/movie/"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzZDU0NTE3ZTZhNDBiMTQ4OGZjNWY3M2JlNGU0NzhjOSIsInN1YiI6IjY0ODM3MDNmZDJiMjA5MDBlYmJmY2IyZiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.lmivo_i1kCuuoJOnvyCnMr1_AaBaNDFnGPjvHOHVGxA"
}

import pandas as pd
import numpy as np

def get_img_urls(link_df, movie_ids):
    
    img_urls = []
    
    for mov_id in movie_ids:
        
        tmdb_id = int(link_df.loc[link_df["movieId"] == mov_id, "tmdbId"])
        mov_url = f"{tmdb_endpoint}{tmdb_id}"
        
        # Make API request
        data = requests.get(mov_url, headers=headers).json()
        
        # Extract the image path from the API response
        try:
            img_url = "https://image.tmdb.org/t/p/original" + data["poster_path"]
        except:
            img_url = "http://www.interlog.com/~tfs/images/posters/TFSMoviePosterUnavailable.jpg"
        
        img_urls.append(img_url)
        
    return img_urls


def get_imdb_links(link_df, movie_ids):
    
    imdb_links = []
    
    for mov_id in movie_ids:
        
        imdb_id = int(link_df.loc[link_df["movieId"] == mov_id, "imdbId"])
        imdb_id = f"{imdb_id}".zfill(7)
        imdb_link = f"https://www.imdb.com/title/tt{imdb_id}"
        imdb_links.append(imdb_link)
    
    return imdb_links


def get_genre_movies(genre, n, mov_df, excl_movies=None):
    
    if excl_movies == None:
        excl_movies = []
        
    genre_sel = mov_df.loc[mov_df["genres"].str.contains(genre) & ~mov_df["movieId"].isin(excl_movies)].sample(n)
    genre_titles = genre_sel["title"].to_list()
    genre_ids = genre_sel["movieId"].to_list()
    excl_movies += genre_ids
    
    return genre_ids, genre_titles, excl_movies


def get_decade_movies(decade, n, mov_df, excl_movies=None):
    
    decade_sel = mov_df.loc[(mov_df["decade"] == decade) & ~mov_df["movieId"].isin(excl_movies)].sample(n)
    decade_titles = decade_sel["title"].to_list()
    decade_ids = decade_sel["movieId"].to_list()
    excl_movies += decade_ids
    
    return decade_ids, decade_titles, excl_movies