import streamlit as st
from recommender_functions_precomputed import *
import pickle
import pandas as pd
st.set_page_config(layout="wide", page_title="WBSFlix")

use_precomuted_item_sim = 0

### Load data (cache to only do this once)
@st.cache_resource
def get_data():
    with open("recommender/data/movie_df.pickle", "rb") as myFile:
        movie_df = pickle.load(myFile)
    with open("recommender/data/rating_df.pickle", "rb") as myFile:
        rating_df = pickle.load(myFile)  
    with open("recommender/data/link_df.pickle", "rb") as myFile:
        link_df = pickle.load(myFile)   
    with open("recommender/data/popularity_df.pickle", "rb") as myFile:
        popularity_df = pickle.load(myFile)
    if use_precomuted_item_sim:
        with open("recommender/data/movie_correlations_min5.pickle", "rb") as myFile:
            movie_correlations = pickle.load(myFile)
    else:
        movie_correlations = []
    with open("recommender/data/predicted_ratings_0.pickle", "rb") as myFile:
        predicted_ratings_0 = pickle.load(myFile)
    with open("recommender/data/predicted_ratings_nan.pickle", "rb") as myFile:
        predicted_ratings_nan = pickle.load(myFile)
    
    return movie_df, rating_df, link_df, popularity_df, movie_correlations, predicted_ratings_0, predicted_ratings_nan

movie_df, rating_df, link_df, popularity_df, movie_correlations, predicted_ratings_0, predicted_ratings_nan = get_data()

### Parameters

# Top n movies to display in each category
n = 8    
img_width = 100


# Style
primaryColor="#fb4b4b"
backgroundColor="#000000"
secondaryBackgroundColor="#313133"
textColor="#ffffff"

    
### Manage user ID
# Check if userID has already been selected
if "userId" not in st.session_state:
    st.session_state["userId"] = "undefined"

if "session_user_id" not in st.session_state:
    st.session_state["session_user_id"] = None
    
if "input_user_id" not in st.session_state:
    st.session_state["input_user_id"] = ""
    
def change_user_state():
    st.session_state["userId"] = "defined"

def clear_user():
    st.session_state["userId"] = "undefined"
    st.session_state["input_user_id"] = ""



### Sidebar
with st.sidebar:
    
    # Title
    st.markdown(f"<h1 style='color:{primaryColor};'>WBSFlix</h1>", unsafe_allow_html=True)
    
    # Input user ID
    label = "What's your user ID?"
    user_id = st.text_input(label, value=st.session_state["input_user_id"], on_change=change_user_state, key="input_user_id")
    
    if st.session_state["userId"] == "defined":
        try:
            if len(user_id) > 0:
                st.session_state["session_user_id"] = int(user_id)

            if st.session_state["session_user_id"] not in predicted_ratings_0.index:
                st.write("Unknown user ID!")
                st.session_state["userId"] = "undefined"
        except:
            st.write("Invalid user ID!")
            st.session_state["userId"] = "undefined"
    
    button_state = st.session_state["userId"] == "undefined"
    
    st.button("Clear user", on_click=clear_user, disabled=button_state, use_container_width=False)
    
    try:
        ref_user = st.session_state["session_user_id"]
    except:
        label = "<span style='color:{primaryColor}:'>Invalid User ID. Please try again</span>"
        
        
### Main body

if st.session_state["userId"] == "undefined":
    
    st.markdown(f"<h1 style='color:{textColor};'>Welcome to <span style='color:{primaryColor};'>WBSFlix</span>!</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:{textColor};'>For personalized recommendations, please enter your User ID.</h4>", unsafe_allow_html=True)
    
    st.divider()
    
    ### Popularity ranking
    
    # Get movies
    excl_movs = None
    top_10_ids_popularity, top_10_titles_popularity = popularity_ranking(n, popularity_df, exclude_movies=excl_movs)
    imgs_popularity = get_img_urls(link_df, top_10_ids_popularity)
    imdb_popularity = get_imdb_links(link_df, top_10_ids_popularity)
    
    # Show movies
    st.subheader("Our customers' favourites...")
    columns = st.columns(n, gap="small")
    for idx, col in enumerate(columns):
        with col:
            st.markdown(f'<a href="{imdb_popularity[idx]}" target="_blank"><img src="{imgs_popularity[idx]}" alt="{top_10_titles_popularity[idx]}" width="{img_width}"></a>', unsafe_allow_html=True)

else:
    
    st.markdown(f"<h1 style='color:{primaryColor};'>We have recommendations for you!</h1>", unsafe_allow_html=True)
    
    st.divider()
    
    ## Popularity ranking
    
    # Get movies
    excl_movs = None
    top_10_ids_popularity, top_10_titles_popularity = popularity_ranking(n, popularity_df, exclude_movies=excl_movs)
    imgs_popularity = get_img_urls(link_df, top_10_ids_popularity)
    imdb_popularity = get_imdb_links(link_df, top_10_ids_popularity)
    
    # Show movies
    st.subheader("Our customers' favourites...")
    columns = st.columns(n, gap="small")
    for idx, col in enumerate(columns):
        with col:
            st.markdown(f'<a href="{imdb_popularity[idx]}" target="_blank"><img src="{imgs_popularity[idx]}" alt="{top_10_titles_popularity[idx]}" width="{img_width}"></a>', unsafe_allow_html=True)

            
    ## User-based collaborative filtering with zero filling
    
    # Get movies
    excl_movs = top_10_ids_popularity
    top_10_ids_user_0, top_10_titles_user_0 = user_based_cf(ref_user, n, predicted_ratings_0, movie_df, exclude_movies=excl_movs)
    imgs_user_0 = get_img_urls(link_df, top_10_ids_user_0)
    imdb_user_0 = get_imdb_links(link_df, top_10_ids_user_0)

    # Show movies
    st.subheader("Popular movies you might like...")
    columns = st.columns(n, gap="small")
    for idx, col in enumerate(columns):
        with col:
            st.markdown(f'<a href="{imdb_user_0[idx]}" target="_blank"><img src="{imgs_user_0[idx]}" alt="{top_10_titles_user_0[idx]}" width="{img_width}"></a>', unsafe_allow_html=True)


    ## User-based collaborative filtering without zero filling
    
    # Get movies
    excl_movs = top_10_ids_popularity + top_10_ids_user_0
    top_10_ids_user_nan, top_10_titles_user_nan = user_based_cf(ref_user, n, predicted_ratings_nan, movie_df, exclude_movies=excl_movs)
    imgs_user_nan = get_img_urls(link_df, top_10_ids_user_nan)
    imdb_user_nan = get_imdb_links(link_df, top_10_ids_user_nan)

    # Show movies
    st.subheader("Special picks just for you...")
    columns = st.columns(n, gap="small")
    for idx, col in enumerate(columns):
        with col:
            st.markdown(f'<a href="{imdb_user_nan[idx]}" target="_blank"><img src="{imgs_user_nan[idx]}" alt="{top_10_titles_user_nan[idx]}" width="{img_width}"></a>', unsafe_allow_html=True)
            
    
    ## Item-based collaborative filtering
    
    # Get favourite movie
    ref_movie_id = rating_df.loc[rating_df["userId"]==ref_user].sort_values(by="rating", ascending=False).head(10).sample(1)["movieId"].values[0]
    ref_movie_title = movie_df.loc[movie_df["movieId"]==ref_movie_id, "title"].values[0]

    # Get similar movies
    excl_movs = top_10_ids_popularity + top_10_ids_user_0 + top_10_ids_user_nan
    if use_precomuted_item_sim:
        top_10_ids_item, top_10_titles_item = item_based_cf(ref_movie_id, n, movie_correlations, movie_df, exclude_movies=excl_movs)
    else:
        top_10_ids_item, top_10_titles_item = item_based_cf_old(ref_movie_id, n, rating_df, movie_df, exclude_movies=excl_movs)
    imgs_item = get_img_urls(link_df, top_10_ids_item)
    imdb_item = get_imdb_links(link_df, top_10_ids_item)

    # Show movies
    st.markdown(f"<h3 style='color:{textColor};'>Because you liked <span style='color:{primaryColor};'>{ref_movie_title}</span>, you might also like...</h3>", unsafe_allow_html=True)
    columns = st.columns(n, gap="small")
    for idx, col in enumerate(columns):
        with col:
            try:
                st.markdown(f'<a href="{imdb_item[idx]}" target="_blank"><img src="{imgs_item[idx]}" alt="{top_10_titles_item[idx]}" width="{img_width}"></a>', unsafe_allow_html=True)
            except:
                pass