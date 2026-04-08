import streamlit as st
import  pickle as pk
import pandas as pd

new_movies_list = pk.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(new_movies_list)
similarity = pk.load(open('similarity.pkl','rb'))

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True, key= lambda x:x[1])[1:6]

    recommend_movies = []
    for i in movies_list:
        movie_id = i[0]
        recommend_movies.append(movies.iloc[i[0]].title)

    return  recommend_movies

st.title('Movie Recommender System')

option = st.selectbox('Select A Movie',movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(option)
    for i in recommendations:
        st.write(i)
