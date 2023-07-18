import pandas as pd
import streamlit as st
from recommender_functions_precomputed import *
import pickle

st.set_page_config(layout="wide")

### Load data (cache to only do this once)
@st.cache_resource
def get_data():
    with open("recommender/data/movie_df.pickle", "rb") as myFile:
        movie_df = pickle.load(myFile)
    with open("recommender/data/link_df.pickle", "rb") as myFile:
        link_df = pickle.load(myFile)  
    
    movie_df['decade'] = movie_df['year'].astype("int").apply(lambda x: 'Ancient Movies' if x < 1950
                                                               else '1950s' if x < 1960
                                                               else '1960s' if x < 1970
                                                               else '1970s' if x < 1980
                                                               else '1980s' if x < 1990
                                                               else '1990s' if x < 2000
                                                               else '2000s' if x < 2010
                                                               else '2010s' if x < 2020
                                                               else '2020s')
    
    return movie_df, link_df

movie_df, link_df = get_data()

### Parameters
n = 8    
img_width = 100

### Create tabs

tab1, tab2 = st.tabs(["Genres", "Decades"])

with tab1:
    
    ## Genres
    genres = ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action',
              'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'Musical', 'Documentary',
               'Western', 'Film-Noir']
    
    rec_list = []
    for genre in genres:

        ids, titles, rec_list = get_genre_movies(genre, n, movie_df, excl_movies=rec_list)
        imgs = get_img_urls(link_df, ids)
        imdb = get_imdb_links(link_df, ids)
        
        st.subheader(genre)
        columns = st.columns(n, gap="small")
        for idx, col in enumerate(columns):
            with col:
                st.markdown(f'<a href="{imdb[idx]}" target="_blank"><img src="{imgs[idx]}" alt="{titles[idx]}" width="{img_width}"></a>', unsafe_allow_html=True)

                
with tab2:
    
    ## Decades
    decades = ["2010s", "2000s", "1990s", "1980s", "1970s", "1960s", "1950s", "Ancient Movies"]

    rec_list = []
    for decade in decades:

        ids, titles, rec_list = get_decade_movies(decade, n, movie_df, excl_movies=rec_list)
        imgs = get_img_urls(link_df, ids)
        imdb = get_imdb_links(link_df, ids)
        
        st.subheader(decade)
        columns = st.columns(n, gap="small")
        for idx, col in enumerate(columns):
            with col:
                st.markdown(f'<a href="{imdb[idx]}" target="_blank"><img src="{imgs[idx]}" alt="{titles[idx]}" width="{img_width}"></a>', unsafe_allow_html=True)
                
                
    