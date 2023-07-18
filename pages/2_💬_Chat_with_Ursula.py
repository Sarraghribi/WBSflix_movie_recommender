import pandas as pd
import streamlit as st
from recommender_functions_precomputed import *
import pickle
from streamlit_chat import message

### Load data (cache to only do this once)
@st.cache_resource
def get_data():
    with open("recommender/data/movie_df.pickle", "rb") as myFile:
        movie_df = pickle.load(myFile)
    with open("recommender/data/rating_df.pickle", "rb") as myFile:
        rating_df = pickle.load(myFile)  
    with open("recommender/data/link_df.pickle", "rb") as myFile:
        link_df = pickle.load(myFile) 
    # with open("data/movie_correlations_min5.pickle", "rb") as myFile:
    #     movie_correlations = pickle.load(myFile)
    
    return movie_df, rating_df, link_df

movie_df, rating_df, link_df = get_data()

### Parameters
n = 5
img_width = 120


### Prepare chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    bot_message = "Hi, I am Ursula. I can help you find movies that you like."
    st.session_state.chat_history.append({"message": bot_message, "is_user": False})
        
    bot_message = "Please type in a movie that you liked. If you can't remember a title, just a keyword is fine, too."
    st.session_state.chat_history.append({"message": bot_message, "is_user": False})
    

if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = 1
    
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
    
if "matching_movies" not in st.session_state:
    st.session_state.matching_movies = []

    
def generate_answer():
    user_message = st.session_state.input_text    
    st.session_state.input_text = ""
    
    if user_message.isdigit():
        st.session_state.chat_stage = 2
    elif len(user_message.strip()) > 0:
        st.session_state.chat_stage = 1
    else:
        return
    
    st.session_state.chat_history.append({"message": user_message, "is_user": True})
    
    # Look up matching movies
    if st.session_state.chat_stage == 1:
        
        matching_movies = movie_df["title"][movie_df["title"].str.contains(user_message, case=False, na=False)].unique()
        st.session_state.matching_movies = matching_movies
        
        if len(matching_movies) > 0:
            
            mov_list = ["Please choose a movie from the following options:\n"]
            for idx, movie in enumerate(matching_movies):
                mov_list.append(f"{idx + 1}. {movie}")
                
            bot_message =  "\n".join(mov_list)
            st.session_state.chat_history.append({"message": bot_message, "is_user": False})
            
            st.session_state.chat_stage = 2
        
        else:
            bot_message = "Sorry, I couldn't find any matching movies based on your input. Please try again with a different keyword."
            st.session_state.chat_history.append({"message": bot_message, "is_user": False})
            
            
    # Get similar movies
    elif st.session_state.chat_stage == 2:  
        
        try:
            test = int(user_message)
        except:
            bot_message = "Invalid choice. Please enter a valid number."
            st.session_state.chat_history.append({"message": bot_message, "is_user": False})
        else:
            
            if int(user_message) not in range(1, len(st.session_state.matching_movies) + 1):

                    bot_message = "Invalid choice. Please enter a valid number."
                    st.session_state.chat_history.append({"message": bot_message, "is_user": False})
            else:
                chosen_movie = st.session_state.matching_movies[int(user_message) - 1]
                chosen_movie_id = movie_df.loc[movie_df["title"]==chosen_movie, "movieId"].values[0]
                # recommended_movie_ids, recommended_movies = item_based_cf(chosen_movie_id, n, movie_correlations, movie_df)
                recommended_movie_ids, recommended_movies = item_based_cf_old(chosen_movie_id, n, rating_df, movie_df)
                
                if len(recommended_movies) > 0:
                    bot_message = "Here are some movies you might like!"
                    st.session_state.chat_history.append({"message": bot_message, "is_user": False})

                    # Print list of movies
                    # mov_list = []
                    # for idx, movie in enumerate(recommended_movies):
                    #     mov_list.append(f"{idx + 1}. {movie}")
                    # bot_message =  "\n".join(mov_list)
                    # st.session_state.chat_history.append({"message": bot_message, "is_user": False})

                    # Show movie posters
                    st.session_state.chat_history.append([recommended_movie_ids, recommended_movies])
                    
                    
                    bot_message = "Enjoy watching!"
                    st.session_state.chat_history.append({"message": bot_message, "is_user": False})
                    
                    bot_message = "You can explore other movies from the list (enter a number) or give me a new keyword if you like!"
                    st.session_state.chat_history.append({"message": bot_message, "is_user": False})

                else:
                    bot_message = "Sorry, I couldn't find any recommendations based on your input. Please choose a different film."
                    st.session_state.chat_history.append({"message": bot_message, "is_user": False})

                    # st.session_state.chat_stage = 0
    
                

def show_movie_posters(link_df, mov_list, img_width):
    
    mov_ids = mov_list[0]
    mov_titles = mov_list[1]
    
    imgs = get_img_urls(link_df, mov_ids)
    imdb = get_imdb_links(link_df, mov_ids)
    
    columns = st.columns(n, gap="small")
    for idx, col in enumerate(columns):
        with col:
            st.markdown(f'<a href="{imdb[idx]}" target="_blank"><img src="{imgs[idx]}" alt="{mov_titles[idx]}" width="{img_width}"></a>', unsafe_allow_html=True)
            st.caption(mov_titles[idx])


for idx, chat in enumerate(st.session_state.chat_history):
    if type(chat) is dict:
        message(**chat, key=str(idx)) 
    elif type(chat) is list:
        show_movie_posters(link_df, chat, img_width)
    
    
### Show text_input
st.text_input("Talk to Ursula", key="input_text", on_change=generate_answer)

