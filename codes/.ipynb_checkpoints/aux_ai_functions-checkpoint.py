# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import json
import os
import glob
import re
import itertools

from sklearn.feature_extraction.text import CountVectorizer
#Import cosine_similarity function
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

def signature_json_to_df(signature_json, genre_solver):

    input_df = pd.DataFrame()
    for username in signature_json['user_signatures'].keys():
        print(username)
        for key in signature_json['user_signatures'][username].keys():
            df_sub = pd.DataFrame(signature_json['user_signatures'][username][key])
            df_sub['username'] = username
            df_sub['duration_key'] = key
            df_sub['party_id'] = signature_json['party_id']
            input_df = input_df.append(df_sub, ignore_index = True)


    for index, row in input_df.iterrows():
        genre_mapping = pd.merge(pd.DataFrame(row.genres, columns = ['genres']), genre_solver, left_on = 'genres', right_on = 'genre')
        if genre_mapping.shape[0] !=0:
            input_df.loc[index, 'category'] = genre_mapping.category.value_counts().index[0]
        else:
            input_df.loc[index, 'category'] = 'unknown'

    signature_df = input_df.loc[input_df.category != 'unknown'].reset_index(drop = True)

    return signature_df

# Function to sanitize data to prevent ambiguity.
# Removes spaces and converts to lowercase
def sanitize(x):
    #Check if x is a list.
    if isinstance(x, list):
        #Strip spaces and convert to lowercase
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Checck if x is a string. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

#Function that creates a soup out of the desired metadata,  Give genre higher weightage
def create_soup(df):
    artists_series = ' '.join(df.artists)
    soup =  (artists_series+' '+df['category']+ ' ' +df['name'])
    return soup


# AI Function
def aux_ai(signature_df, selected_genres):

    #Apply the generate_list function to cast, keywords, director and genres
    for feature in ['name','artists', 'category']:
        signature_df[feature] = signature_df[feature].apply(sanitize)

    # Creating the metadata soup
    signature_df_copy = signature_df.copy()

    # Create the new soup feature
    signature_df_copy['soup'] = signature_df_copy.apply(create_soup, axis=1)

    #Define a new CountVectorizer object and create vectors for the soup
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(signature_df_copy['soup'])

    #Compute the cosine similarity score (equivalent to dot product for tf-idf vectors)
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    similarity_df = pd.DataFrame(cosine_sim)
    # similarity_df = similarity_df.replace(0,np.nan)

    main_df = pd.concat([signature_df_copy, similarity_df ], axis =1 )

    user_means = main_df.groupby(['username']).mean().T

    # Average Satifaction score across all users
    mean_df = main_df.groupby(['username']).mean()

    final_scores = pd.DataFrame(stats.hmean(mean_df), columns = ['final score'])
    final_scores.sort_values('final score', ascending = False, inplace = True)

    indices = main_df.loc[main_df.category.isin(selected_genres)].index

    selected_scores = final_scores.loc[final_scores.index.isin(indices)]
    selected_scores.sort_values('final score', ascending = False, inplace = True) #Double Check ascending sort

    final_df = signature_df.iloc[selected_scores.index]
    unique_df = final_df.drop_duplicates(['name'])
    unique_df = unique_df.astype('str').groupby(['artists']).head(3)

    N = 25
    l = len(selected_genres)
    ai_top_n = pd.DataFrame()
    for category in selected_genres:
        print(category)
        x = int(N/l)
        df_sub = unique_df.loc[unique_df.category == category,:].head(x)
        ai_top_n = ai_top_n.append(df_sub)

# ai_top_df = ai_top_n.loc[:,['name','artists','category']].reset_index(drop = True)
    return ai_top_n.to_dict(orient = 'records')